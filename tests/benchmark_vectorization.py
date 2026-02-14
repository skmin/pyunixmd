#!/usr/bin/env python
"""
Benchmark script for CTv2 vectorization schemes.

This script compares the performance of loop-based implementations vs
vectorized implementations for 6 key functions in ctv2.py.

Usage:
    python benchmark_vectorization.py [--ntrajs N] [--nst N] [--nat_qm N] [--ndim N] [--repeat N]
"""

import numpy as np
import time
import argparse
from typing import Callable, Tuple


def timer(func: Callable, *args, repeat: int = 100) -> Tuple[float, any]:
    """Time a function over multiple runs and return average time and result."""
    # Warmup
    result = func(*args)

    start = time.perf_counter()
    for _ in range(repeat):
        result = func(*args)
    end = time.perf_counter()

    return (end - start) / repeat, result


class BenchmarkData:
    """Container for benchmark test data."""

    def __init__(self, ntrajs: int = 50, nst: int = 3, nat_qm: int = 10, ndim: int = 3):
        self.ntrajs = ntrajs
        self.nst = nst
        self.nat_qm = nat_qm
        self.ndim = ndim
        self.nst_pair = int(nst * (nst - 1) / 2)
        self.small = 1e-8

        # Random seed for reproducibility
        np.random.seed(42)

        # Generate test data
        self.rho_diag = np.random.rand(ntrajs, nst)
        self.rho_diag /= self.rho_diag.sum(axis=1, keepdims=True)  # Normalize

        self.l_coh = np.random.rand(ntrajs, nst) > 0.3  # ~70% coherent
        self.l_first = np.zeros((ntrajs, nst), dtype=bool)

        self.phase = np.random.randn(ntrajs, nst, nat_qm, ndim)
        self.qmom = np.random.randn(ntrajs, nat_qm, ndim)
        self.qmom_bo = np.random.randn(ntrajs, self.nst_pair, nat_qm, ndim)

        self.d2S = np.random.randn(ntrajs, nst, nat_qm)
        self.alpha = np.random.rand(self.nst_pair, nat_qm)
        self.beta = np.random.rand(self.nst_pair, nat_qm)

        self.pos = np.random.randn(ntrajs, nat_qm, ndim)
        self.sigma = np.abs(np.random.randn(nst, nat_qm, ndim)) + 0.1
        self.avg_R = np.random.randn(nst, nat_qm, ndim)

        self.g_i_I = np.random.rand(nst, ntrajs) + 0.1
        self.g_i_IJ = np.random.rand(nst, ntrajs, ntrajs) + 0.01
        self.g_I = np.sum(self.g_i_I, axis=0)

        self.pseudo_pop = np.random.rand(nst, ntrajs)
        self.pseudo_pop /= self.pseudo_pop.sum(axis=0, keepdims=True)

        self.inv_mass = 1.0 / (np.random.rand(nat_qm) + 0.5)

        self.upper_th = 0.99
        self.lower_th = 0.01


# =============================================================================
# Case 1: check_coherence - rho extraction
# =============================================================================

def check_coherence_loop(data: BenchmarkData) -> np.ndarray:
    """Original loop-based rho extraction."""
    rho_tmp = np.zeros((data.ntrajs, data.nst))
    for itraj in range(data.ntrajs):
        for ist in range(data.nst):
            rho_tmp[itraj, ist] = data.rho_diag[itraj, ist]
    return rho_tmp


def check_coherence_vectorized(data: BenchmarkData) -> np.ndarray:
    """Vectorized rho extraction."""
    return data.rho_diag.copy()


# =============================================================================
# Case 2: calculate_qmom - K calculation
# =============================================================================

def calculate_K_loop(data: BenchmarkData) -> np.ndarray:
    """Original loop-based K calculation."""
    K = np.zeros((data.ntrajs, data.nst, data.nst))

    for itraj in range(data.ntrajs):
        index_lk = 0
        for ist in range(data.nst):
            for jst in range(ist + 1, data.nst):
                if data.l_coh[itraj, ist] and data.l_coh[itraj, jst]:
                    phase_diff = data.phase[itraj, ist] - data.phase[itraj, jst]
                    qmom_phase = np.sum(data.qmom[itraj] * phase_diff, axis=1)
                    K[itraj, ist, jst] = 0.5 * np.sum(data.inv_mass * qmom_phase)
                    K[itraj, jst, ist] = -K[itraj, ist, jst]
                index_lk += 1
    return K


def calculate_K_vectorized(data: BenchmarkData) -> np.ndarray:
    """Vectorized K calculation."""
    K = np.zeros((data.ntrajs, data.nst, data.nst))

    # Vectorized phase difference: (ntrajs, nst, nst, nat_qm, ndim)
    phase_diff = data.phase[:, :, np.newaxis, :, :] - data.phase[:, np.newaxis, :, :, :]

    # Coherence mask for state pairs: (ntrajs, nst, nst)
    coh_ij = data.l_coh[:, :, np.newaxis] & data.l_coh[:, np.newaxis, :]

    # qmom_phase: (ntrajs, nst, nst, nat_qm)
    qmom_phase = np.einsum('tad,tijad->tija', data.qmom, phase_diff)

    # K: (ntrajs, nst, nst)
    K_full = 0.5 * np.einsum('a,tija->tij', data.inv_mass, qmom_phase)

    # Apply coherence mask and upper triangular
    triu_mask = np.triu(np.ones((data.nst, data.nst), dtype=bool), k=1)
    K = np.where(coh_ij & triu_mask, K_full, 0.0)
    K -= K.transpose(0, 2, 1)

    return K


# =============================================================================
# Case 3: calculate_slope with l_traj_gaussian
# =============================================================================

def calculate_gaussian_loop(data: BenchmarkData) -> Tuple[np.ndarray, np.ndarray]:
    """Original loop-based Gaussian calculation for trajectory-centered Gaussians."""
    g_i_IJ = np.ones((data.nst, data.ntrajs, data.ntrajs))
    g_i_I = np.ones((data.nst, data.ntrajs))

    for ist in range(data.nst):
        sigma_sq = data.sigma[ist] ** 2
        norm_factor = np.prod(1.0 / np.sqrt(2.0 * np.pi * sigma_sq))

        for itraj in range(data.ntrajs):
            g_i_I[ist, itraj] = 0.0
            for jtraj in range(data.ntrajs):
                pos_diff = data.pos[itraj] - data.pos[jtraj]
                gauss_val = np.exp(-0.5 * np.sum(pos_diff ** 2 / sigma_sq))
                g_i_IJ[ist, itraj, jtraj] = gauss_val * norm_factor * data.rho_diag[jtraj, ist] / data.ntrajs
                g_i_I[ist, itraj] += g_i_IJ[ist, itraj, jtraj]

    return g_i_I, g_i_IJ


def calculate_gaussian_vectorized(data: BenchmarkData) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Gaussian calculation for trajectory-centered Gaussians."""
    g_i_IJ = np.ones((data.nst, data.ntrajs, data.ntrajs))
    g_i_I = np.ones((data.nst, data.ntrajs))

    # pos_diff: (ntrajs, ntrajs, nat_qm, ndim)
    pos_diff = data.pos[:, np.newaxis, :, :] - data.pos[np.newaxis, :, :, :]

    for ist in range(data.nst):
        sigma_sq = data.sigma[ist] ** 2

        # Gaussian exponent: (ntrajs, ntrajs)
        gauss_exp = -0.5 * np.sum(pos_diff ** 2 / sigma_sq[np.newaxis, np.newaxis, :, :], axis=(2, 3))
        gauss_val = np.exp(gauss_exp)

        norm_factor = np.prod(1.0 / np.sqrt(2.0 * np.pi * sigma_sq))

        g_i_IJ[ist, :, :] = gauss_val * norm_factor * data.rho_diag[:, ist][np.newaxis, :] / data.ntrajs
        g_i_I[ist, :] = np.sum(g_i_IJ[ist, :, :], axis=1)

    return g_i_I, g_i_IJ


# =============================================================================
# Case 4: calculate_center with l_traj_gaussian
# =============================================================================

def calculate_intercept_loop(data: BenchmarkData) -> np.ndarray:
    """Original loop-based intercept calculation for trajectory-centered Gaussians."""
    intercept = np.zeros((data.ntrajs, data.nat_qm, data.ndim))
    sigma_sq = data.sigma ** 2
    inv_sigma_sq = np.where(sigma_sq > data.small, 1.0 / sigma_sq, 0.0)

    for itraj in range(data.ntrajs):
        if data.g_I[itraj] / data.ntrajs >= data.small:
            for iat in range(data.nat_qm):
                for isp in range(data.ndim):
                    val = 0.0
                    for jtraj in range(data.ntrajs):
                        val += np.sum(data.g_i_IJ[:, itraj, jtraj] * data.pos[jtraj, iat, isp] * inv_sigma_sq[:, iat, isp])
                    intercept[itraj, iat, isp] = -val / data.g_I[itraj]

    return intercept


def calculate_intercept_vectorized(data: BenchmarkData) -> np.ndarray:
    """Vectorized intercept calculation for trajectory-centered Gaussians."""
    intercept = np.zeros((data.ntrajs, data.nat_qm, data.ndim))
    sigma_sq = data.sigma ** 2
    inv_sigma_sq = np.where(sigma_sq > data.small, 1.0 / sigma_sq, 0.0)

    valid_g_I = data.g_I / data.ntrajs >= data.small

    # Using einsum: 'sij,jad,sad->iad'
    intercept_sum = np.einsum('sij,jad,sad->iad', data.g_i_IJ, data.pos, inv_sigma_sq)

    g_I_safe = np.where(valid_g_I, data.g_I, 1.0)
    intercept = np.where(valid_g_I[:, np.newaxis, np.newaxis],
                         -intercept_sum / g_I_safe[:, np.newaxis, np.newaxis],
                         0.0)

    return intercept


# =============================================================================
# Case 5: set_avg_pop_cons - atom loop
# =============================================================================

def set_avg_pop_cons_loop(data: BenchmarkData) -> np.ndarray:
    """Original loop-based numer calculation with atom loop."""
    numer = np.zeros((data.nst_pair, data.nat_qm))

    index_lk = 0
    for ist in range(data.nst):
        for jst in range(ist + 1, data.nst):
            coh_mask = data.l_coh[:, ist] & data.l_coh[:, jst]
            rho_prod = data.rho_diag[:, ist] * data.rho_diag[:, jst]
            weight = rho_prod * coh_mask

            for iat in range(data.nat_qm):
                phase_diff = data.phase[:, ist, iat, :] - data.phase[:, jst, iat, :]
                qmom_diff = data.qmom_bo[:, index_lk, iat, :] - data.qmom[:, iat, :]
                qmom_phase = np.sum(qmom_diff * phase_diff, axis=1)
                numer[index_lk, iat] = -np.sum(qmom_phase * weight)

            index_lk += 1

    return numer


def set_avg_pop_cons_vectorized(data: BenchmarkData) -> np.ndarray:
    """Vectorized numer calculation without atom loop."""
    numer = np.zeros((data.nst_pair, data.nat_qm))

    index_lk = 0
    for ist in range(data.nst):
        for jst in range(ist + 1, data.nst):
            coh_mask = data.l_coh[:, ist] & data.l_coh[:, jst]
            rho_prod = data.rho_diag[:, ist] * data.rho_diag[:, jst]
            weight = rho_prod * coh_mask

            # Vectorized over atoms
            phase_diff = data.phase[:, ist, :, :] - data.phase[:, jst, :, :]
            qmom_diff = data.qmom_bo[:, index_lk, :, :] - data.qmom
            qmom_phase = np.sum(qmom_diff * phase_diff, axis=2)
            numer[index_lk, :] = -np.sum(qmom_phase * weight[:, np.newaxis], axis=0)

            index_lk += 1

    return numer


# =============================================================================
# Case 6: calculate_force - Laplacian/beta terms
# =============================================================================

def calculate_lap_force_loop(data: BenchmarkData) -> np.ndarray:
    """Original loop-based Laplacian force calculation."""
    ctforce = np.zeros((data.nat_qm, data.ndim))
    phase = data.phase[0]  # Use first trajectory
    rho_diag = data.rho_diag[0]

    index_lk = 0
    for ist in range(data.nst):
        for jst in range(ist + 1, data.nst):
            d2S_diff = data.d2S[0, ist, :] - data.d2S[0, jst, :]
            lap_term = np.sum(data.inv_mass * d2S_diff * data.alpha[index_lk, :])
            ctforce -= 2.0 * lap_term * (phase[ist] - phase[jst]) * rho_diag[ist] * rho_diag[jst]
            index_lk += 1

    return ctforce


def calculate_lap_force_vectorized(data: BenchmarkData) -> np.ndarray:
    """Vectorized Laplacian force calculation."""
    ctforce = np.zeros((data.nat_qm, data.ndim))
    phase = data.phase[0]
    rho_diag = data.rho_diag[0]

    # phase_diff: (nst, nst, nat_qm, ndim)
    phase_diff = phase[:, np.newaxis, :, :] - phase[np.newaxis, :, :, :]
    rho_ij = rho_diag[:, np.newaxis] * rho_diag[np.newaxis, :]

    # d2S_diff: (nst, nst, nat_qm)
    d2S_diff = data.d2S[0, :, np.newaxis, :] - data.d2S[0, np.newaxis, :, :]

    # Build alpha matrix
    alpha_mat = np.zeros((data.nst, data.nst, data.nat_qm))
    index_lk = 0
    for ist in range(data.nst):
        for jst in range(ist + 1, data.nst):
            alpha_mat[ist, jst, :] = data.alpha[index_lk, :]
            alpha_mat[jst, ist, :] = data.alpha[index_lk, :]
            index_lk += 1

    # lap_term: (nst, nst)
    lap_term = np.einsum('a,ija,ija->ij', data.inv_mass, d2S_diff, alpha_mat)

    triu_mask = np.triu(np.ones((data.nst, data.nst), dtype=bool), k=1)
    lap_force = np.einsum('ij,ijkl,ij->kl', np.where(triu_mask, lap_term, 0.0), phase_diff, rho_ij)
    ctforce -= 2.0 * lap_force

    return ctforce


# =============================================================================
# Benchmark runner
# =============================================================================

def run_benchmark(ntrajs: int = 50, nst: int = 3, nat_qm: int = 10, ndim: int = 3, repeat: int = 100):
    """Run all benchmarks and print results."""

    print("=" * 80)
    print("CTv2 Vectorization Benchmark")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  ntrajs  = {ntrajs}")
    print(f"  nst     = {nst}")
    print(f"  nat_qm  = {nat_qm}")
    print(f"  ndim    = {ndim}")
    print(f"  repeat  = {repeat}")
    print()

    data = BenchmarkData(ntrajs, nst, nat_qm, ndim)

    benchmarks = [
        ("1. check_coherence (rho extraction)",
         check_coherence_loop, check_coherence_vectorized),
        ("2. calculate_qmom (K calculation)",
         calculate_K_loop, calculate_K_vectorized),
        ("3. calculate_slope (l_traj_gaussian)",
         calculate_gaussian_loop, calculate_gaussian_vectorized),
        ("4. calculate_center (l_traj_gaussian)",
         calculate_intercept_loop, calculate_intercept_vectorized),
        ("5. set_avg_pop_cons (atom loop)",
         set_avg_pop_cons_loop, set_avg_pop_cons_vectorized),
        ("6. calculate_force (Laplacian term)",
         calculate_lap_force_loop, calculate_lap_force_vectorized),
    ]

    print("-" * 80)
    print(f"{'Benchmark':<45} {'Loop (ms)':<12} {'Vector (ms)':<12} {'Speedup':<10} {'Match'}")
    print("-" * 80)

    total_loop_time = 0
    total_vec_time = 0

    for name, loop_func, vec_func in benchmarks:
        # Time both implementations
        loop_time, loop_result = timer(loop_func, data, repeat=repeat)
        vec_time, vec_result = timer(vec_func, data, repeat=repeat)

        # Convert to milliseconds
        loop_ms = loop_time * 1000
        vec_ms = vec_time * 1000

        total_loop_time += loop_ms
        total_vec_time += vec_ms

        # Calculate speedup
        speedup = loop_time / vec_time if vec_time > 0 else float('inf')

        # Check if results match
        if isinstance(loop_result, tuple):
            match = all(np.allclose(l, v, rtol=1e-10) for l, v in zip(loop_result, vec_result))
        else:
            match = np.allclose(loop_result, vec_result, rtol=1e-10)

        match_str = "OK" if match else "FAIL"

        print(f"{name:<45} {loop_ms:>10.4f}   {vec_ms:>10.4f}   {speedup:>8.2f}x  {match_str}")

    print("-" * 80)
    total_speedup = total_loop_time / total_vec_time if total_vec_time > 0 else float('inf')
    print(f"{'TOTAL':<45} {total_loop_time:>10.4f}   {total_vec_time:>10.4f}   {total_speedup:>8.2f}x")
    print("=" * 80)

    # Memory estimate
    print(f"\nMemory estimates for vectorized arrays:")
    print(f"  phase_diff (ntrajs,nst,nst,nat_qm,ndim): {ntrajs*nst*nst*nat_qm*ndim*8/1024/1024:.2f} MB")
    print(f"  pos_diff (ntrajs,ntrajs,nat_qm,ndim):    {ntrajs*ntrajs*nat_qm*ndim*8/1024/1024:.2f} MB")
    print(f"  g_i_IJ (nst,ntrajs,ntrajs):              {nst*ntrajs*ntrajs*8/1024/1024:.2f} MB")


def run_scaling_benchmark(repeat: int = 50):
    """Run scaling benchmarks for different problem sizes."""

    print("\n" + "=" * 80)
    print("Scaling Benchmark")
    print("=" * 80)

    # Test scaling with number of trajectories
    print("\n--- Scaling with ntrajs (nst=3, nat_qm=10) ---")
    print(f"{'ntrajs':<10} {'Loop (ms)':<12} {'Vector (ms)':<12} {'Speedup'}")
    print("-" * 50)

    for ntrajs in [10, 25, 50, 100, 200]:
        data = BenchmarkData(ntrajs=ntrajs, nst=3, nat_qm=10, ndim=3)

        loop_time, _ = timer(calculate_gaussian_loop, data, repeat=repeat)
        vec_time, _ = timer(calculate_gaussian_vectorized, data, repeat=repeat)

        speedup = loop_time / vec_time if vec_time > 0 else float('inf')
        print(f"{ntrajs:<10} {loop_time*1000:>10.4f}   {vec_time*1000:>10.4f}   {speedup:>8.2f}x")

    # Test scaling with number of atoms
    print("\n--- Scaling with nat_qm (ntrajs=50, nst=3) ---")
    print(f"{'nat_qm':<10} {'Loop (ms)':<12} {'Vector (ms)':<12} {'Speedup'}")
    print("-" * 50)

    for nat_qm in [5, 10, 20, 50, 100]:
        data = BenchmarkData(ntrajs=50, nst=3, nat_qm=nat_qm, ndim=3)

        loop_time, _ = timer(calculate_intercept_loop, data, repeat=repeat)
        vec_time, _ = timer(calculate_intercept_vectorized, data, repeat=repeat)

        speedup = loop_time / vec_time if vec_time > 0 else float('inf')
        print(f"{nat_qm:<10} {loop_time*1000:>10.4f}   {vec_time*1000:>10.4f}   {speedup:>8.2f}x")

    # Test scaling with number of states
    print("\n--- Scaling with nst (ntrajs=50, nat_qm=10) ---")
    print(f"{'nst':<10} {'Loop (ms)':<12} {'Vector (ms)':<12} {'Speedup'}")
    print("-" * 50)

    for nst in [2, 3, 4, 5, 6]:
        data = BenchmarkData(ntrajs=50, nst=nst, nat_qm=10, ndim=3)

        loop_time, _ = timer(calculate_K_loop, data, repeat=repeat)
        vec_time, _ = timer(calculate_K_vectorized, data, repeat=repeat)

        speedup = loop_time / vec_time if vec_time > 0 else float('inf')
        print(f"{nst:<10} {loop_time*1000:>10.4f}   {vec_time*1000:>10.4f}   {speedup:>8.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CTv2 vectorization schemes")
    parser.add_argument("--ntrajs", type=int, default=50, help="Number of trajectories")
    parser.add_argument("--nst", type=int, default=3, help="Number of states")
    parser.add_argument("--nat_qm", type=int, default=10, help="Number of QM atoms")
    parser.add_argument("--ndim", type=int, default=3, help="Number of dimensions")
    parser.add_argument("--repeat", type=int, default=100, help="Number of repetitions for timing")
    parser.add_argument("--scaling", action="store_true", help="Run scaling benchmarks")

    args = parser.parse_args()

    run_benchmark(args.ntrajs, args.nst, args.nat_qm, args.ndim, args.repeat)

    if args.scaling:
        run_scaling_benchmark(args.repeat)
