from __future__ import division
from lib.libctmqcv2 import el_run
from mqc.mqc import MQC
from misc import eps, au_to_K, au_to_A, call_name, typewriter, gaussian1d, close_files
import os, shutil, textwrap
import numpy as np
import pickle

class CTv2(MQC):
    """ Class for coupled-trajectory mixed quantum-classical (CTMQC) dynamics

        :param object,list molecules: List for molecule objects
        :param object thermostat: Thermostat object
        :param integer,list istates: List for initial state
        :param double dt: Time interval
        :param integer nsteps: Total step of nuclear propagation
        :param integer nesteps: Total step of electronic propagation
        :param string elec_object: Electronic equation of motions
        :param string propagator: Electronic propagator
        :param boolean l_print_dm: Logical to print BO population and coherence
        :param boolean l_adj_nac: Adjust nonadiabatic coupling to align the phases
        :param double rho_threshold: Electronic density threshold for decoherence term calculation
        :param init_coefs: Initial BO coefficient
        :type init_coefs: double, 2D list or complex, 2D list
        :param string unit_dt: Unit of time interval
        :param integer out_freq: Frequency of printing output
        :param integer verbosity: Verbosity of output
        :param boolean l_crunch: Perform CTv2
        :param boolean l_dc_w_mom: Use state-wise momentum for the phase term
        :param boolean l_traj_gaussian: Use the sum of trajectory centered Gaussians for the nuclear density
        :param integer t_cons: Average population conservation scheme. 0: none, 1: Scaling (only for l_lap=True), 2: Shift
        :param boolean l_etot0: Use the constant total energy (at t=0) for the state-wise momentum calculation
        :param boolean l_lap: Include laplacian ENC term
        :param boolean l_en_cons: Adjust momentum at every time step to enforce the total energy conservation
        :param double artifact_expon: Exponent for width for nuclear density estimation (used only when l_traj_gaussian = True)
        :param boolean l_asymp: Terminate dynamics when the trajectory reaches asymptotic region (use this option for model systems only)
        :param double x_fin: Define asymptotic region (a.u.)
        :param boolean l_real_pop: Use |C_j|^2 for |chi_j|^2/|chi|^2 in quantum momentum calculation.
        :param integer t_pc: Phase correction scheme (1: use P, 2: use sum_j nabla S_j)
    """
    def __init__(self, molecules, thermostat=None, istates=None, dt=0.5, nsteps=1000, nesteps=20, \
        elec_object="coefficient", propagator="rk4", l_print_dm=True, l_adj_nac=True, rho_threshold=0.01, \
        init_coefs=None, unit_dt="fs", out_freq=1, verbosity=0, \
        l_crunch=True, l_dc_w_mom=True, l_traj_gaussian=False, \
        t_cons=2, l_etot0=True, l_lap=False,\
        l_en_cons=False, artifact_expon=0.2, l_asymp=False, x_fin=25.0, \
        l_real_pop=True, t_pc=1):
        # Save name of MQC dynamics
        self.md_type = self.__class__.__name__

        # Initialize input values
        self.mols = molecules
        self.ntrajs = len(self.mols)
        self.digit = len(str(self.ntrajs))

        self.nst = self.mols[0].nst
        self.nat_qm = self.mols[0].nat_qm
        self.ndim = self.mols[0].ndim

        # Check compatibility between istates and init_coefs
        self.istates = istates
        self.init_coefs = init_coefs
        self.check_istates()

        # Initialize input values and coefficient for first trajectory
        super().__init__(self.mols[0], thermostat, self.istates[0], dt, nsteps, nesteps, \
            elec_object, propagator, l_print_dm, l_adj_nac, self.init_coefs[0], unit_dt, out_freq, verbosity)

        # Exception for electronic propagation
        if (self.elec_object != "coefficient"):
            error_message = "Electronic equation motion in CTMQC is only solved with respect to coefficient!"
            error_vars = f"elec_object = {self.elec_object}"
            raise NotImplementedError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        # Exception for thermostat
        if (self.thermo != None):
            error_message = "Thermostat is not implemented yet!"
            error_vars = f"thermostat = {self.thermo}"
            raise NotImplementedError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

        # Initialize coefficient for other trajectories
        for itraj in range(1, self.ntrajs):
            self.mols[itraj].get_coefficient(self.init_coefs[itraj], self.istates[itraj])

        # Initialize variables for CTMQC
        self.phase = np.zeros((self.ntrajs, self.nst, self.nat_qm, self.ndim)) # phase term, same as self.dS if l_dc_w_mom=True
        self.nst_pair = int(self.nst * (self.nst - 1) / 2) # number of state pair
        self.qmom = np.zeros((self.ntrajs, self.nat_qm, self.ndim)) # total quantum momentum: \mathcal{P}_\nu = \nabla_\nu|\chi|^2 / |\chi|^2
        self.qmom_bo = np.zeros((self.ntrajs, self.nst_pair, self.nat_qm, self.ndim)) # projected quantum momentum: \mathcal{G}_{\nu,ij} =\nabla_\nu|\chi_i|^2 / |\chi_i|^2 + \nabla_\nu|\chi_j|^2 / |\chi_j|^2
        self.K = np.zeros((self.ntrajs, self.nst, self.nst)) # Decoherence term from total quantum momentum: \sum_\nu 1/(2M_\nu) \mathcal{P}_\nu \cdot \nabla_\nu (S_i - S_j)
        self.K_bo = np.zeros((self.ntrajs, self.nst, self.nst)) # Decoherence term from projected quantum momentum: \sum_\nu 1/(2M_\nu) \mathcal{G}_{\nu,ij} \cdot \nabla_\nu (S_i - S_j)
        self.mom = np.zeros((self.ntrajs, self.nst, self.nat_qm, self.ndim)) # state-wise momentum
        self.dS = np.zeros((self.ntrajs, self.nst, self.nat_qm, self.ndim)) # phase term used in phase correction
        self.d2e = np.zeros((self.ntrajs, self.nst, self.nat_qm)) # Laplacian of BO energy
        self.d2S = np.zeros((self.ntrajs, self.nst, self.nat_qm)) # Laplacian of S_i's 
        self.alpha = np.zeros((self.nst_pair, self.nat_qm)) # Scaling factor
        self.beta = np.zeros((self.nst_pair, self.nat_qm)) # Shift factor
        self.etot0 = np.zeros((self.ntrajs)) # Total energy at t=0
        self.l_coh = np.zeros((self.ntrajs, self.nst), dtype=bool) # Flag that a trajectory is in coherence
        self.l_first = np.zeros((self.ntrajs, self.nst), dtype=bool) # Flag that a trajectory become coherent

        # Initialize variables to calculate quantum momentum 
        self.count_ntrajs = np.ones((self.ntrajs, self.nat_qm, self.ndim)) * self.ntrajs
        self.sigma = np.zeros((self.nst, self.nat_qm, self.ndim)) 
        self.slope = np.zeros((self.ntrajs, self.nat_qm, self.ndim))
        self.intercept = np.zeros((self.ntrajs, self.nat_qm, self.ndim))
        self.slope_bo = np.zeros((self.ntrajs, self.nst_pair, self.nat_qm, self.ndim))
        self.intercept_bo = np.zeros((self.ntrajs, self.nst_pair, self.nat_qm, self.ndim))
        self.g_I = np.zeros((self.ntrajs)) # |\chi|^2
        self.g_i_I = np.ones((self.nst, self.ntrajs)) # |\chi_i|^2
        self.g_i_IJ = np.ones((self.nst, self.ntrajs, self.ntrajs)) # Gaussian basis for |\chi_i|^2 when l_traj_gaussian=True
        self.center = np.zeros((self.ntrajs, self.nat_qm, self.ndim))
        self.center_bo = np.zeros((self.ntrajs, self.nst_pair, self.nat_qm, self.ndim))
        self.avg_R = np.zeros((self.nst, self.nat_qm, self.ndim))
        self.pseudo_pop = np.zeros((self.nst, self.ntrajs))

        # Determine parameters to calculate decoherenece effect
        self.small = 1.0E-08

        self.rho_threshold = rho_threshold
        self.upper_th = 1. - self.rho_threshold
        self.lower_th = self.rho_threshold

        self.artifact_expon = artifact_expon

        self.l_en_cons = l_en_cons
        self.dotpopnac = np.zeros((self.ntrajs, self.nst))
        self.dotpopdec = np.zeros((self.ntrajs, self.nst))
        
        self.l_crunch = l_crunch
        self.l_real_pop = l_real_pop
        self.l_dc_w_mom = l_dc_w_mom
        self.l_traj_gaussian = l_traj_gaussian
        self.l_lap = l_lap
        self.t_pc = t_pc
        self.t_cons = t_cons
        self.l_etot0 = l_etot0

        # Variables for aborting dynamics when all trajectories reach asymptotic region
        self.l_asymp = l_asymp
        self.x_fin = x_fin
        
        # Initialize event to print
        self.event = {"DECO": []}

    def run(self, qm, mm=None, output_dir="./", l_save_qm_log=False, l_save_mm_log=False, l_save_scr=True, restart=None):
        """ Run MQC dynamics according to CTMQC dynamics

            :param object qm: QM object containing on-the-fly calculation infomation
            :param object mm: MM object containing MM calculation infomation
            :param string output_dir: Name of directory where outputs to be saved.
            :param boolean l_save_qm_log: Logical for saving QM calculation log
            :param boolean l_save_mm_log: Logical for saving MM calculation log
            :param boolean l_save_scr: Logical for saving scratch directory
            :param string restart: Option for controlling dynamics restarting
        """
        # Initialize PyUNIxMD
        qm.calc_coupling = True
        qm.calc_tdp = False
        qm.calc_tdp_grad = False
        abs_path_output_dir = os.path.join(os.getcwd(), output_dir)
        base_dirs, unixmd_dirs, samp_bin_dirs, qm_log_dirs, mm_log_dirs = \
            self.run_init(qm, mm, output_dir, False, False, l_save_qm_log, l_save_mm_log, \
            l_save_scr, restart)

        bo_list = [ist for ist in range(self.nst)]
        qm.calc_coupling = True

        self.print_init(qm, mm, restart)

        if (restart == None):
            # Calculate initial input geometry for all trajectories at t = 0.0 s
            self.istep = -1
            for itraj in range(self.ntrajs):
                self.mol = self.mols[itraj]

                self.mol.reset_bo(qm.calc_coupling)
                qm.get_data(self.mol, base_dirs[itraj], bo_list, self.dt, self.istep, calc_force_only=False)

                # TODO: QM/MM
                self.mol.get_nacme()
                
                self.check_decoherence(itraj)
                self.check_coherence(itraj)
                
                self.update_energy()
                
                self.get_state_mom(itraj)
                
                self.get_phase(itraj)

            if (self.t_pc != 0):
                self.get_dS()

            self.calculate_qmom()
            
            if (self.l_lap):
                self.get_d2S()
            
            self.set_avg_pop_cons()

            for itraj in range(self.ntrajs):

                self.mol = self.mols[itraj]

                self.write_md_output(itraj, unixmd_dirs[itraj], qm.calc_coupling, self.istep)

                self.print_step(self.istep, itraj)

        #TODO: restart
        elif (restart == "write"):
            # Reset initial time step to t = 0.0 s
            self.istep = -1
            for itraj in range(self.ntrajs):
                self.write_md_output(itraj, unixmd_dirs[itraj], qm.calc_coupling, self.istep)
                self.print_step(self.istep, itraj)

        elif (restart == "append"):
            # Set initial time step to last successful step of previous dynamics
            self.istep = self.fstep

        self.istep += 1

        # Main MD loop
        for istep in range(self.istep, self.nsteps):
            for itraj in range(self.ntrajs):
                self.mol = self.mols[itraj]

                self.calculate_force(itraj)
                self.cl_update_position()

                self.mol.backup_bo(qm.calc_coupling)
                self.mol.reset_bo(qm.calc_coupling)

                qm.get_data(self.mol, base_dirs[itraj], bo_list, self.dt, self.istep, calc_force_only=False)

                if (not self.mol.l_nacme and self.l_adj_nac):
                    self.mol.adjust_nac()

                #TODO: QM/MM

                self.calculate_force(itraj)
                self.cl_update_velocity()

                self.mol.get_nacme()
                
                self.update_energy()

                el_run(self, itraj)

                #TODO: thermostat
                #if (self.thermo != None):
                #    self.thermo.run(self)

                self.check_decoherence(itraj)
                self.check_coherence(itraj)
                
                self.update_energy()

                self.get_state_mom(itraj)
                
                self.get_phase(itraj)
            
            if (self.t_pc != 0):
                self.get_dS()
            
            self.calculate_qmom()
            
            if (self.l_lap):
                self.get_d2S()
            
            self.set_avg_pop_cons()
            
            for itraj in range(self.ntrajs):
                self.mol = self.mols[itraj]

                if ((istep + 1) % self.out_freq == 0):
                    self.write_md_output(itraj, unixmd_dirs[itraj], qm.calc_coupling, istep)
                    self.print_step(istep, itraj)
                if (istep == self.nsteps - 1):
                    self.write_final_xyz(unixmd_dirs[itraj], istep)

            self.fstep = istep
            #restart_file = os.path.join(abs_path_output_dir, "RESTART.bin")
            #with open(restart_file, 'wb') as f:
            #    pickle.dump({'qm':qm, 'md':self}, f)

            for itraj in range(self.ntrajs):
                l_abort = True
                det = self.mols[itraj].pos[0, 0] * self.mols[itraj].vel[0, 0]
                if (self.l_asymp and det > 0. and np.abs(self.mols[itraj].pos[0, 0]) > np.abs(self.x_fin)):
                    pass
                else:
                    l_abort = False
                    break
            
            if(l_abort):
                break

        # Close open file handles for all trajectory directories
        for itraj in range(self.ntrajs):
            close_files(unixmd_dirs[itraj])

        # Delete scratch directory
        if (not l_save_scr):
            for itraj in range(self.ntrajs):
                tmp_dir = os.path.join(unixmd_dirs[itraj], "scr_qm")
                if (os.path.exists(tmp_dir)):
                    shutil.rmtree(tmp_dir)
    
    def get_dS(self):
        self.dS[:, :, :, :] = self.mom[:, :, :, :]

    def get_d2S(self):
        # Vectorized: Calculate d2S by integrating Laplacian of BO energy
        self.d2S -= self.d2e * self.dt
            
    def set_avg_pop_cons(self):

        # Get rho diagonal for all trajectories
        rho_diag = np.array([np.diag(mol.rho.real) for mol in self.mols])  # (ntrajs, nst)

        # Rescale d2S
        if (self.t_cons == 1 and self.l_lap):
            deno = np.zeros((self.nst_pair, self.nat_qm))
            numer = np.zeros((self.nst_pair, self.nat_qm))
            index_lk = 0
            for ist in range(self.nst):
                for jst in range(ist + 1, self.nst):
                    # Coherence mask and rho products
                    coh_mask = self.l_coh[:, ist] & self.l_coh[:, jst]  # (ntrajs,)
                    rho_prod = rho_diag[:, ist] * rho_diag[:, jst]  # (ntrajs,)
                    weight = rho_prod * coh_mask  # (ntrajs,)

                    # Vectorized over atoms: phase_diff (ntrajs, nat_qm, ndim)
                    phase_diff = self.phase[:, ist, :, :] - self.phase[:, jst, :, :]  # (ntrajs, nat_qm, ndim)

                    if (self.l_crunch):
                        qmom_diff = self.qmom_bo[:, index_lk, :, :] - self.qmom  # (ntrajs, nat_qm, ndim)
                    else:
                        qmom_diff = self.qmom  # (ntrajs, nat_qm, ndim)

                    # qmom_phase: sum over ndim -> (ntrajs, nat_qm)
                    qmom_phase = np.sum(qmom_diff * phase_diff, axis=2)
                    # numer: weighted sum over trajectories -> (nat_qm,)
                    numer[index_lk, :] = -np.sum(qmom_phase * weight[:, np.newaxis], axis=0)

                    # d2S_diff: (ntrajs, nat_qm)
                    d2S_diff = self.d2S[:, ist, :] - self.d2S[:, jst, :]
                    deno[index_lk, :] = np.sum(d2S_diff * weight[:, np.newaxis], axis=0)

                    self.alpha[index_lk, :] = np.where(
                        np.abs(deno[index_lk, :]) >= self.small,
                        numer[index_lk, :] / deno[index_lk, :],
                        0.0
                    )
                    index_lk += 1

        # Or add const: beta (\Delta_{ij} in the paper and note)
        elif (self.t_cons == 2):
            self.alpha[:, :] = 1.0
            deno = np.zeros((self.nst_pair))
            numer = np.zeros((self.nst_pair, self.nat_qm))
            index_lk = 0
            for ist in range(self.nst):
                for jst in range(ist + 1, self.nst):
                    # Coherence mask and rho products
                    coh_mask = self.l_coh[:, ist] & self.l_coh[:, jst]  # (ntrajs,)
                    rho_prod = rho_diag[:, ist] * rho_diag[:, jst]  # (ntrajs,)
                    weight = rho_prod * coh_mask  # (ntrajs,)

                    deno[index_lk] = np.sum(rho_prod)

                    if (np.abs(deno[index_lk]) < self.small):
                        self.beta[index_lk, :] = 0.0
                    else:
                        # Vectorized over atoms: phase_diff (ntrajs, nat_qm, ndim)
                        phase_diff = self.phase[:, ist, :, :] - self.phase[:, jst, :, :]

                        if (self.l_crunch):
                            qmom_diff = self.qmom_bo[:, index_lk, :, :] - self.qmom
                        else:
                            qmom_diff = self.qmom

                        # qmom_phase: sum over ndim -> (ntrajs, nat_qm)
                        qmom_phase = np.sum(qmom_diff * phase_diff, axis=2)
                        numer[index_lk, :] = -np.sum(qmom_phase * weight[:, np.newaxis], axis=0)

                        if (self.l_lap):
                            # d2S_diff: (ntrajs, nat_qm)
                            d2S_diff = self.d2S[:, ist, :] - self.d2S[:, jst, :]
                            numer[index_lk, :] -= np.sum(d2S_diff * weight[:, np.newaxis], axis=0)

                        self.beta[index_lk, :] = numer[index_lk, :] / deno[index_lk]
                    index_lk += 1
        else:
            self.alpha[:, :] = 1.0
            self.beta[:, :] = 0.0

    def get_state_mom(self, itrajectory):

        if (self.istep == -1):
            self.etot0[itrajectory] = self.mol.etot

        # Vectorized state momentum calculation
        energies = np.array([st.energy for st in self.mol.states])  # (nst,)
        if (self.l_etot0):
            alpha = (self.etot0[itrajectory] - energies) / self.mol.ekin
        else:
            alpha = (self.mol.etot - energies) / self.mol.ekin

        alpha = np.maximum(alpha, 0.)  # Clip negative values
        sqrt_alpha = np.sqrt(alpha)  # (nst,)

        # vel: (nat_qm, ndim), mass: (nat_qm,)
        # mom shape: (nst, nat_qm, ndim)
        vel_mass = self.mol.vel[:self.nat_qm, :] * self.mol.mass[:self.nat_qm, np.newaxis]  # (nat_qm, ndim)
        self.mom[itrajectory, :, :, :] = sqrt_alpha[:, np.newaxis, np.newaxis] * vel_mass[np.newaxis, :, :]

    def calculate_force(self, itrajectory):
        """ Routine to calculate force

            :param integer itrajectory: Index for trajectories
        """
        self.rforce = np.zeros((self.nat_qm, self.ndim))

        # Derivatives of energy
        rho_diag = np.diag(self.mol.rho.real)  # (nst,)
        for ist, istate in enumerate(self.mols[itrajectory].states):
            self.rforce += istate.force * rho_diag[ist]

        # Non-adiabatic forces
        energies = np.array([st.energy for st in self.mol.states])  # (nst,)
        for ist in range(self.nst):
            for jst in range(ist + 1, self.nst):
                self.rforce += 2. * self.mol.nac[ist, jst] * self.mol.rho.real[ist, jst] \
                    * (energies[ist] - energies[jst])

        # CT forces = -\sum_{i,j} |C_iC_j|^2 K_lk_{i, j} * (f_i - f_j)
        ctforce = np.zeros((self.nat_qm, self.ndim))
        rho_ij = rho_diag[:, np.newaxis] * rho_diag[np.newaxis, :]  # (nst, nst)
        phase = self.phase[itrajectory]  # (nst, nat_qm, ndim)
        phase_diff = phase[:, np.newaxis, :, :] - phase[np.newaxis, :, :, :]  # (nst, nst, nat_qm, ndim)

        if (self.l_crunch):
            # Vectorized CT force with CRUNCH
            K_diff = self.K_bo[itrajectory] - self.K[itrajectory]  # (nst, nst)
            ctforce = -np.einsum('ij,ijkl,ij->kl', K_diff, phase_diff, rho_ij)

            inv_mass = 1. / self.mol.mass[0:self.nat_qm]  # (nat_qm,)

            if (self.l_lap):
                # Vectorized Laplacian term
                # d2S_diff: (nst, nst, nat_qm) = d2S[ist] - d2S[jst]
                d2S_diff = self.d2S[itrajectory, :, np.newaxis, :] - self.d2S[itrajectory, np.newaxis, :, :]

                # Build alpha matrix (nst, nst, nat_qm) from state-pair indexed alpha
                alpha_mat = np.zeros((self.nst, self.nst, self.nat_qm))
                index_lk = 0
                for ist in range(self.nst):
                    for jst in range(ist + 1, self.nst):
                        alpha_mat[ist, jst, :] = self.alpha[index_lk, :]
                        alpha_mat[jst, ist, :] = self.alpha[index_lk, :]
                        index_lk += 1

                # lap_term: (nst, nst) = sum over atoms of (inv_mass * d2S_diff * alpha)
                lap_term = np.einsum('a,ija,ija->ij', inv_mass, d2S_diff, alpha_mat)

                # ctforce contribution: 2.0 * lap_term[i,j] * phase_diff[i,j] * rho_diag[i] * rho_diag[j]
                # Only upper triangular matters due to antisymmetry
                triu_mask = np.triu(np.ones((self.nst, self.nst), dtype=bool), k=1)
                lap_force = np.einsum('ij,ijkl,ij->kl', np.where(triu_mask, lap_term, 0.0), phase_diff, rho_ij)
                ctforce -= 2.0 * lap_force

            if (self.t_cons == 2):
                # Vectorized Beta term
                # Build beta_term: (nst, nst) = sum over atoms of (inv_mass * beta)
                beta_mat = np.zeros((self.nst, self.nst))
                index_lk = 0
                for ist in range(self.nst):
                    for jst in range(ist + 1, self.nst):
                        beta_mat[ist, jst] = np.sum(inv_mass * self.beta[index_lk, :])
                        beta_mat[jst, ist] = beta_mat[ist, jst]
                        index_lk += 1

                # ctforce contribution
                triu_mask = np.triu(np.ones((self.nst, self.nst), dtype=bool), k=1)
                beta_force = np.einsum('ij,ijkl,ij->kl', np.where(triu_mask, beta_mat, 0.0), phase_diff, rho_ij)
                ctforce -= 2.0 * beta_force
        else:
            # Vectorized CT force without CRUNCH
            ctforce = -np.einsum('ij,ijkl,ij->kl', self.K[itrajectory], phase_diff, rho_ij)

        # Finally, force is Ehrenfest force + CT force
        self.rforce += ctforce

    def update_energy(self):
        """ Routine to update the energy of molecules in CTMQC dynamics
        """
        # Update kinetic energy
        self.mol.update_kinetic()

        # Vectorized potential energy calculation
        energies = np.array([st.energy for st in self.mol.states])
        rho_diag = np.diag(self.mol.rho.real)
        self.mol.epot = np.dot(rho_diag, energies)

        if (self.l_en_cons and not (self.istep == -1)):
            alpha = (self.mol.etot - self.mol.epot)
            factor = alpha / self.mol.ekin

            self.mol.vel *= np.sqrt(factor)
            self.mol.update_kinetic()

        self.mol.etot = self.mol.epot + self.mol.ekin

    def get_phase(self, itrajectory):
        """ Routine to calculate phase

            :param integer itrajectory: Index for trajectories
        """
        if (self.l_dc_w_mom):
            # Vectorized: copy all states at once
            self.phase[itrajectory, :, :, :] = self.mom[itrajectory, :, :, :]
        else:
            # Vectorized: accumulate forces for all states
            forces = np.array([st.force for st in self.mol.states])  # (nst, nat_qm, ndim)
            self.phase[itrajectory, :, :, :] += forces * self.dt

    def check_coherence(self, itrajectory):
        """ Routine to check coherence among BO states

            :param integer itrajectory: Index for trajectories
        """
        # Vectorized rho extraction
        rho_tmp = np.array([np.diag(mol.rho.real) for mol in self.mols])  # (ntrajs, nst)

        # Vectorized average and trajectory rho
        avg_rho = np.mean(rho_tmp, axis=0)  # (nst,)
        rho = rho_tmp[itrajectory]  # (nst,)

        # Vectorized coherence check
        out_of_bounds = ((rho > self.upper_th) | (rho < self.lower_th) |
                         (avg_rho > self.upper_th) | (avg_rho < self.lower_th))

        # Store previous coherence state for l_first calculation
        was_coherent = self.l_coh[itrajectory].copy()

        # Update coherence flags
        self.l_coh[itrajectory] = ~out_of_bounds

        # Calculate l_first: True if newly coherent (was not coherent, now is)
        newly_coherent = self.l_coh[itrajectory] & ~was_coherent
        self.l_first[itrajectory] = newly_coherent

        # Count coherent states
        count = np.sum(self.l_coh[itrajectory])

        if count < 2:
            self.l_coh[itrajectory, :] = False
            self.l_first[itrajectory, :] = False
            newly_coherent = np.zeros(self.nst, dtype=bool)

        # Build event string for newly coherent states
        new_states = np.where(newly_coherent)[0]
        if len(new_states) > 0:
            tmp_st = ", ".join(str(ist) for ist in new_states)
            self.event["DECO"].append(f"{itrajectory + 1:8d}: {tmp_st} states are in coherence")

    def check_decoherence(self, itrajectory):
        """ Routine to check decoherence among BO states

            :param integer itrajectory: Index for trajectories
        """
        for ist in range(self.mol.nst):
            if (self.l_coh[itrajectory, ist]):
                rho = self.mol.rho.real[ist, ist]
                if (rho > self.upper_th):
                    #self.set_decoherence(ist)
                    self.event["DECO"].append(f"{itrajectory + 1:8d}: decohered to {ist} state")
                    return
    
    def set_decoherence(self, one_st):
        """ Routine to reset coefficient/density if the state is decohered

            :param integer one_st: State index that its population is one
        """
        self.mol.rho = np.zeros((self.mol.nst, self.mol.nst), dtype=np.complex64)
        self.mol.rho[one_st, one_st] = 1. + 0.j
        
        if (self.elec_object == "coefficient"):
            for ist in range(self.mol.nst):
                if (ist == one_st):
                    self.mol.states[ist].coef /= np.absolute(self.mol.states[ist].coef).real
                else:
                    self.mol.states[ist].coef = 0. + 0.j

    def calculate_qmom(self):
        """ Routine to calculate quantum momentum
        """
        # _lk means state_pair dependency.
        # i and j are trajectory index.
        # -------------------------------------------------------------------
        # 1. Calculate variances for each trajectory
        self.calculate_sigma()

        # 2. Calculate slope
        self.calculate_slope()

        # 3. Calculate the center of quantum momentum
        self.calculate_center()

        # 4. Compute quantum momentum (vectorized)
        # G_{\nu, ij} = (\nabla_\nu|\chi_i|^2 / |\chi_i|^2  + \nabla_\nu|\chi_j|^2/|\chi_j|^2)
        # and/or
        # P_{\nu} = \nabla_\nu|\chi|^2 / |\chi|^2
        pos = np.array([mol.pos for mol in self.mols])  # (ntrajs, nat_qm, ndim)
        self.qmom = self.slope * pos - self.intercept

        if (self.l_crunch):
            # qmom_bo: (ntrajs, nst_pair, nat_qm, ndim)
            self.qmom_bo = self.slope_bo * pos[:, np.newaxis, :, :] - self.intercept_bo

        # 5. Calculate K and K_bo (fully vectorized)
        # K_bo = 0.5 * G_{\nu, ij}/M \cdot D_{ij}
        # and/or
        # K = 0.5 * P_{\nu}/M \cdot D_{ij}
        self.K.fill(0.)
        self.K_bo.fill(0.)

        inv_mass = 1. / self.mol.mass[0:self.nat_qm]  # (nat_qm,)

        # Vectorized phase difference: (ntrajs, nst, nst, nat_qm, ndim)
        phase_diff = self.phase[:, :, np.newaxis, :, :] - self.phase[:, np.newaxis, :, :, :]

        # Coherence mask for state pairs: (ntrajs, nst, nst)
        coh_ij = self.l_coh[:, :, np.newaxis] & self.l_coh[:, np.newaxis, :]

        # qmom_phase: sum over ndim of qmom * phase_diff -> (ntrajs, nst, nst, nat_qm)
        # qmom: (ntrajs, nat_qm, ndim), phase_diff: (ntrajs, nst, nst, nat_qm, ndim)
        qmom_phase = np.einsum('tad,tijad->tija', self.qmom, phase_diff)

        # K: sum over atoms with inv_mass -> (ntrajs, nst, nst)
        K_full = 0.5 * np.einsum('a,tija->tij', inv_mass, qmom_phase)

        # Apply coherence mask and upper triangular
        triu_mask = np.triu(np.ones((self.nst, self.nst), dtype=bool), k=1)
        self.K = np.where(coh_ij & triu_mask, K_full, 0.0)
        self.K -= self.K.transpose(0, 2, 1)  # Make antisymmetric

        if self.l_crunch:
            # Build index mapping for state pairs
            index_lk = 0
            for ist in range(self.nst):
                for jst in range(ist + 1, self.nst):
                    # qmom_bo: (ntrajs, nst_pair, nat_qm, ndim)
                    qmom_bo_phase = np.sum(self.qmom_bo[:, index_lk, :, :] * phase_diff[:, ist, jst, :, :], axis=2)  # (ntrajs, nat_qm)
                    K_bo_val = 0.5 * np.sum(inv_mass * qmom_bo_phase, axis=1)  # (ntrajs,)
                    self.K_bo[:, ist, jst] = np.where(coh_ij[:, ist, jst], K_bo_val, 0.0)
                    self.K_bo[:, jst, ist] = -self.K_bo[:, ist, jst]
                    index_lk += 1


    def calculate_sigma(self):
        """ Routine to calculate variances for each trajectories
        """
        # Vectorized data extraction
        pos = np.array([mol.pos for mol in self.mols])  # (ntrajs, nat_qm, ndim)
        rho = np.array([np.diag(mol.rho.real) for mol in self.mols])  # (ntrajs, nst)

        # Vectorized sigma calculation for each state
        rho_sum = np.sum(rho, axis=0)  # (nst,)
        rho_avg = rho_sum / self.ntrajs  # (nst,)

        for ist in range(self.nst):
            if (rho_avg[ist] < self.lower_th):
                self.avg_R[ist, :, :] = 0.0
                self.sigma[ist, :, :] = np.inf
            else:
                self.avg_R[ist, :, :] = np.tensordot(pos, rho[:, ist], axes=(0, 0)) / rho_sum[ist]
                self.sigma[ist, :, :] = np.sqrt(np.tensordot(pos ** 2, rho[:, ist], axes=(0, 0)) / rho_sum[ist] - self.avg_R[ist, :, :] ** 2)

        if (self.l_traj_gaussian):
            for ist in range(self.nst):
                if (rho_avg[ist] < self.lower_th):
                    self.avg_R[ist, :, :] = 0.0
                    self.sigma[ist, :, :] = np.inf
                else:
                    self.sigma[ist, :, :] *= 1.06 * (rho_avg[ist] * self.ntrajs) ** (-self.artifact_expon)



    def calculate_slope(self):
        """ Routine to calculate slope
        """
        # Vectorized data extraction
        pos = np.array([mol.pos for mol in self.mols])  # (ntrajs, nat_qm, ndim)
        rho = np.array([np.diag(mol.rho.real) for mol in self.mols])  # (ntrajs, nst)

        self.g_I.fill(0.)
        self.g_i_I.fill(1.)
        self.g_i_IJ.fill(1.)

        rho_avg = np.sum(rho, axis=0) / self.ntrajs  # (nst,)

        for ist in range(self.nst):
            if (rho_avg[ist] < self.lower_th):
                self.g_i_I[ist, :] = 0.0
            else:
                if (self.l_traj_gaussian):
                    # Fully vectorized Gaussian calculation for trajectory-centered Gaussians
                    # pos_diff: (ntrajs, ntrajs, nat_qm, ndim) = pos[i] - pos[j]
                    pos_diff = pos[:, np.newaxis, :, :] - pos[np.newaxis, :, :, :]
                    sigma_sq = self.sigma[ist] ** 2  # (nat_qm, ndim)

                    # Gaussian exponent: sum over atoms and dims of (pos_diff^2 / sigma_sq)
                    # Result: (ntrajs, ntrajs)
                    gauss_exp = -0.5 * np.sum(pos_diff ** 2 / sigma_sq[np.newaxis, np.newaxis, :, :], axis=(2, 3))
                    gauss_val = np.exp(gauss_exp)  # (ntrajs, ntrajs)

                    norm_factor = np.prod(1.0 / np.sqrt(2.0 * np.pi * sigma_sq))

                    # g_i_IJ[ist, itraj, jtraj] = gauss_val[itraj, jtraj] * norm_factor * rho[jtraj, ist] / ntrajs
                    self.g_i_IJ[ist, :, :] = gauss_val * norm_factor * rho[:, ist][np.newaxis, :] / self.ntrajs

                    # g_i_I[ist, itraj] = sum over jtraj of g_i_IJ[ist, itraj, jtraj]
                    self.g_i_I[ist, :] = np.sum(self.g_i_IJ[ist, :, :], axis=1)
                else:
                    # Vectorized Gaussian calculation for all trajectories
                    pos_diff = pos - self.avg_R[ist]  # (ntrajs, nat_qm, ndim)
                    sigma_sq = self.sigma[ist] ** 2  # (nat_qm, ndim)
                    # Gaussian product over all atoms and dims: exp(-0.5 * sum((x-mu)^2/sigma^2))
                    gauss_val = np.exp(-0.5 * np.sum(pos_diff ** 2 / sigma_sq[np.newaxis, :, :], axis=(1, 2)))  # (ntrajs,)
                    norm_factor = np.prod(1.0 / np.sqrt(2.0 * np.pi * sigma_sq))
                    self.g_i_I[ist, :] = gauss_val * norm_factor * rho_avg[ist]

        self.g_I[:] = np.sum(self.g_i_I, axis=0)  # |\chi|^2

        # Vectorized pseudo_pop calculation
        self.pseudo_pop.fill(0.)
        if (self.l_real_pop):
            self.pseudo_pop = rho.T  # (nst, ntrajs)
        else:
            valid_mask = self.g_I >= self.small
            self.pseudo_pop[:, valid_mask] = self.g_i_I[:, valid_mask] / self.g_I[valid_mask]

        # Vectorized slope calculation: slope = -sum_ist(pseudo_pop[ist] / sigma[ist]^2)
        # pseudo_pop: (nst, ntrajs), sigma: (nst, nat_qm, ndim)
        sigma_sq = self.sigma ** 2  # (nst, nat_qm, ndim)
        inv_sigma_sq = np.where(sigma_sq > self.small, 1.0 / sigma_sq, 0.0)  # (nst, nat_qm, ndim)
        # slope: (ntrajs, nat_qm, ndim) = -sum over ist of pseudo_pop[ist, itraj] * inv_sigma_sq[ist, iat, isp]
        self.slope = -np.einsum('st,sab->tab', self.pseudo_pop, inv_sigma_sq)

        if (self.l_crunch):
            # Vectorized slope_bo calculation
            index_lk = 0
            for ist in range(self.nst):
                for jst in range(ist + 1, self.nst):
                    sigma_i_sq = self.sigma[ist] ** 2  # (nat_qm, ndim)
                    sigma_j_sq = self.sigma[jst] ** 2  # (nat_qm, ndim)
                    # Check for small sigmas
                    valid = (sigma_i_sq >= self.small) & (sigma_j_sq >= self.small)
                    slope_val = np.where(valid, -(1.0 / sigma_i_sq + 1.0 / sigma_j_sq), 0.0)
                    self.slope_bo[:, index_lk, :, :] = slope_val[np.newaxis, :, :]
                    index_lk += 1
        
    def calculate_center(self):
        """ Routine to calculate center or intercept of quantum momentum
        """
        # Vectorized data extraction
        pos = np.array([mol.pos for mol in self.mols])  # (ntrajs, nat_qm, ndim)

        # Vectorized intercept calculation
        sigma_sq = self.sigma ** 2  # (nst, nat_qm, ndim)
        inv_sigma_sq = np.where(sigma_sq > self.small, 1.0 / sigma_sq, 0.0)  # (nst, nat_qm, ndim)

        if (self.l_traj_gaussian):
            # Fully vectorized for trajectory-centered Gaussians
            valid_g_I = self.g_I / self.ntrajs >= self.small  # (ntrajs,)

            # intercept = -sum_jtraj sum_ist (g_i_IJ[ist,itraj,jtraj] * pos[jtraj] / sigma[ist]^2) / g_I[itraj]
            # g_i_IJ: (nst, ntrajs, ntrajs), pos: (ntrajs, nat_qm, ndim), inv_sigma_sq: (nst, nat_qm, ndim)

            # Weighted position: g_i_IJ[s,i,j] * pos[j,a,d] * inv_sigma_sq[s,a,d]
            # Sum over j and s to get intercept[i,a,d]
            # Using einsum: 'sij,jad,sad->iad'
            intercept_sum = np.einsum('sij,jad,sad->iad', self.g_i_IJ, pos, inv_sigma_sq)

            # Safe division by g_I
            g_I_safe = np.where(valid_g_I, self.g_I, 1.0)
            self.intercept = np.where(valid_g_I[:, np.newaxis, np.newaxis],
                                       -intercept_sum / g_I_safe[:, np.newaxis, np.newaxis],
                                       0.0)

            # For invalid trajectories, center = pos
            self.center = np.where(valid_g_I[:, np.newaxis, np.newaxis], self.center, pos)
        else:
            # Vectorized: intercept = -sum_ist(pseudo_pop[ist] * avg_R[ist] / sigma[ist]^2)
            # pseudo_pop: (nst, ntrajs), avg_R: (nst, nat_qm, ndim), inv_sigma_sq: (nst, nat_qm, ndim)
            weighted_avg_R = self.avg_R * inv_sigma_sq  # (nst, nat_qm, ndim)
            self.intercept = -np.einsum('st,sab->tab', self.pseudo_pop, weighted_avg_R)

        # Calculate center from slope and intercept (safe division)
        slope_valid = np.abs(self.slope) >= self.small
        slope_safe = np.where(slope_valid, self.slope, 1.0)  # Avoid division by zero
        self.center = np.where(slope_valid, self.intercept / slope_safe, pos)

        if (self.l_crunch):
            # Vectorized intercept_bo calculation
            index_lk = 0
            for ist in range(self.nst):
                for jst in range(ist + 1, self.nst):
                    sigma_i_sq = self.sigma[ist] ** 2  # (nat_qm, ndim)
                    sigma_j_sq = self.sigma[jst] ** 2  # (nat_qm, ndim)
                    valid_sigma = (sigma_i_sq >= self.small) & (sigma_j_sq >= self.small)

                    if (self.l_traj_gaussian):
                        # Fully vectorized for trajectory-centered Gaussians with l_crunch
                        # g_i_valid: (ntrajs,) - valid if both g_i_I[ist] and g_i_I[jst] are above threshold
                        g_i_valid = ((self.g_i_I[ist, :] / self.ntrajs >= self.small) &
                                     (self.g_i_I[jst, :] / self.ntrajs >= self.small))

                        # Combined validity mask
                        all_sigma_valid = np.all(valid_sigma)

                        if all_sigma_valid:
                            # Safe division factors
                            g_i_I_ist_safe = np.where(g_i_valid, self.g_i_I[ist, :], 1.0)  # (ntrajs,)
                            g_i_I_jst_safe = np.where(g_i_valid, self.g_i_I[jst, :], 1.0)  # (ntrajs,)

                            # Term 1: sum_j (g_i_IJ[ist,i,j] * pos[j,a,d] / sigma_i_sq[a,d]) / g_i_I[ist,i]
                            # g_i_IJ[ist]: (ntrajs, ntrajs), pos: (ntrajs, nat_qm, ndim)
                            term1_sum = np.einsum('ij,jad->iad', self.g_i_IJ[ist], pos) / sigma_i_sq[np.newaxis, :, :]
                            term1 = term1_sum / g_i_I_ist_safe[:, np.newaxis, np.newaxis]

                            # Term 2: sum_j (g_i_IJ[jst,i,j] * pos[j,a,d] / sigma_j_sq[a,d]) / g_i_I[jst,i]
                            term2_sum = np.einsum('ij,jad->iad', self.g_i_IJ[jst], pos) / sigma_j_sq[np.newaxis, :, :]
                            term2 = term2_sum / g_i_I_jst_safe[:, np.newaxis, np.newaxis]

                            intercept_val = -(term1 + term2)

                            # Apply validity mask
                            self.intercept_bo[:, index_lk, :, :] = np.where(
                                g_i_valid[:, np.newaxis, np.newaxis] & valid_sigma[np.newaxis, :, :],
                                intercept_val,
                                0.0
                            )
                        else:
                            self.intercept_bo[:, index_lk, :, :] = 0.0

                        # Update slope_bo for invalid cases
                        self.slope_bo[:, index_lk, :, :] = np.where(
                            g_i_valid[:, np.newaxis, np.newaxis] & valid_sigma[np.newaxis, :, :],
                            self.slope_bo[:, index_lk, :, :],
                            0.0
                        )
                    else:
                        # Vectorized: intercept_bo = -(avg_R[ist]/sigma[ist]^2 + avg_R[jst]/sigma[jst]^2)
                        inv_sigma_i_sq = np.where(sigma_i_sq >= self.small, 1.0 / sigma_i_sq, 0.0)
                        inv_sigma_j_sq = np.where(sigma_j_sq >= self.small, 1.0 / sigma_j_sq, 0.0)
                        intercept_val = -(self.avg_R[ist] * inv_sigma_i_sq + self.avg_R[jst] * inv_sigma_j_sq)
                        self.intercept_bo[:, index_lk, :, :] = np.where(valid_sigma[np.newaxis, :, :], intercept_val[np.newaxis, :, :], 0.0)

                    # Calculate center_bo
                    slope_bo_valid = np.abs(self.slope_bo[:, index_lk, :, :]) >= self.small
                    self.center_bo[:, index_lk, :, :] = np.where(
                        slope_bo_valid,
                        self.intercept_bo[:, index_lk, :, :] / self.slope_bo[:, index_lk, :, :],
                        pos
                    )
                    index_lk += 1

    def check_istates(self):
        """ Routine to check istates and init_coefs
        """
        if (self.istates != None):
            if (isinstance(self.istates, list)):
                if (len(self.istates) != self.ntrajs):
                    error_message = "Number of elements of initial states must be equal to number of trajectories!"
                    error_vars = f"len(istates) = {len(self.istates)}, ntrajs = {self.ntrajs}"
                    raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")
                else:
                    self.init_coefs = [None] * self.ntrajs
            else:
                error_message = "The type of initial states must be list!"
                error_vars = f"istates = {self.istates}"
                raise TypeError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")
        else:
            if (self.init_coefs == None):
                error_message = "Either initial states or coefficients must be given!"
                error_vars = f"istates = {self.istates}, init_coefs = {self.init_coefs}"
                raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")
            else:
                if (isinstance(self.init_coefs, list)):
                    if (len(self.init_coefs) != self.ntrajs):
                        error_message = "Number of elements of initial coefficients must be equal to number of trajectories!"
                        error_vars = f"len(init_coefs) = {len(self.init_coefs)}, ntrajs = {self.ntrajs}"
                        raise ValueError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")
                    else:
                        self.istates = [None] * self.ntrajs
                else:
                    error_message = "Type of initial coefficients must be list!"
                    error_vars = f"init_coefs = {self.init_coefs}"
                    raise TypeError (f"( {self.md_type}.{call_name()} ) {error_message} ( {error_vars} )")

    def write_md_output(self, itrajectory, unixmd_dir, calc_coupling, istep):
        """ Write output files

            :param integer itrajectory: Index for trajectories
            :param string unixmd_dir: PyUNIxMD directory
            :param boolean calc_coupling: Logical to calculate coupling terms
            :param integer istep: Current MD step
        """
        # Write the common part
        super().write_md_output(unixmd_dir, calc_coupling, istep)

        # Write time-derivative BO population
        self.write_dotpop(itrajectory, unixmd_dir, istep)

        # Write decoherence information
        self.write_dec(itrajectory, unixmd_dir, istep)

    def write_dotpop(self, itrajectory, unixmd_dir, istep):
        """ Write time-derivative BO population

            :param integer itrajectory: Index for trajectories
            :param string unixmd_dir: PyUNIxMD directory
            :param integer istep: Current MD step
        """
        if (self.verbosity >= 1):
            # Write NAC term in DOTPOPNAC
            tmp = f'{istep + 1:9d}' + "".join([f'{pop:15.8f}' for pop in self.dotpopnac[itrajectory]])
            typewriter(tmp, unixmd_dir, "DOTPOPNAC", "a")

            # Write decoherence term in DOTPOPDEC
            tmp = f'{istep + 1:9d}' + "".join([f'{pop:15.8f}' for pop in self.dotpopdec[itrajectory]])
            typewriter(tmp, unixmd_dir, "DOTPOPDEC", "a")

    def write_dec(self, itrajectory, unixmd_dir, istep):
        """ Write CT-based decoherence information

            :param integer itrajectory: Index for trajectories
            :param string unixmd_dir: PyUNIxMD directory
            :param integer istep: Current MD step
        """
        if (self.verbosity >= 1):
            # Write K_lk
            for ist in range(self.nst):
                for jst in range(self.nst):
                    if (ist != jst):
                        tmp = f'{istep + 1:9d}{self.K[itrajectory, ist, jst]:15.8f}'
                        typewriter(tmp, unixmd_dir, f"K_{ist}_{jst}", "a")
                        
                        tmp = f'{istep + 1:9d}{self.K_bo[itrajectory, ist, jst]:15.8f}'
                        typewriter(tmp, unixmd_dir, f"K_BO_{ist}_{jst}", "a")

        # Write detailed quantities related to decoherence
        if (self.verbosity >= 2):
            tmp = f'{istep + 1:9d}' + "".join([f'{pop:15.8f}' for pop in self.pseudo_pop[:, itrajectory]])
            typewriter(tmp, unixmd_dir, "PSEUDOPOP", "a")
            if (itrajectory == 0):
                for ist in range(self.nst):
                    tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}sigma_x{"":5s}sigma_y{"":5s}sigma_z{"":5s}' + \
                        "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                        "".join([f'{self.sigma[ist, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
                    typewriter(tmp, unixmd_dir, f"SIGMA_{ist}", "a")
                    
                    tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s} Center (au)' + \
                        "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                        "".join([f'{self.avg_R[ist, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
                    typewriter(tmp, unixmd_dir, f"CENTER_{ist}", "a")
                    
                    #tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}sigma_x{"":5s}sigma_y{"":5s}sigma_z{"":5s}count_ntrajs' + \
                    #    "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                    #    "".join([f'{self.sigma[itrajectory, ist, iat, idim]:15.8f}' for idim in range(self.ndim)]) + \
                    #    "".join([f'{self.count_ntrajs[itrajectory, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
                    #typewriter(tmp, unixmd_dir, f"SIGMA", "a")
            
            # Write quantum momenta
            tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Momentum (au)' + \
                "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                "".join([f'{self.qmom[itrajectory, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
            typewriter(tmp, unixmd_dir, f"QMOM", "a")
            tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Slope' + \
                "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                "".join([f'{self.slope[itrajectory, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
            typewriter(tmp, unixmd_dir, f"SLOPE", "a")
            
            tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Intercept' + \
                "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                "".join([f'{self.intercept[itrajectory, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
            typewriter(tmp, unixmd_dir, f"INTERCEPT", "a")

            index_lk = -1
            for ist in range(self.nst):
                for jst in range(ist + 1, self.nst):
                    index_lk += 1
                    #tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Momentum center (au)' + \
                    #    "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                    #    "".join([f'{self.center_lk[itrajectory, index_lk, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
                    #typewriter(tmp, unixmd_dir, f"CENTER_{ist}_{jst}", "a")
                    
                    tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Momentum (au)' + \
                        "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                        "".join([f'{self.qmom_bo[itrajectory, index_lk, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
                    typewriter(tmp, unixmd_dir, f"QMOM_BO_{ist}_{jst}", "a")
                    
                    tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Slope' + \
                        "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                        "".join([f'{self.slope_bo[itrajectory, index_lk, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
                    typewriter(tmp, unixmd_dir, f"SLOPE_BO_{ist}_{jst}", "a")
                    
                    tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Intercept' + \
                        "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                        "".join([f'{self.intercept_bo[itrajectory, index_lk, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
                    typewriter(tmp, unixmd_dir, f"INTERCEPT_BO_{ist}_{jst}", "a")

            # Write Phase
            for ist in range(self.mol.nst):
                tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Phase (au)' + \
                    "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                    "".join([f'{self.phase[itrajectory, ist, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
                typewriter(tmp, unixmd_dir, f"PHASE_{ist}", "a")
            
            # Write state momentum
            for ist in range(self.mol.nst):
                tmp = f'{self.nat_qm:6d}\n{"":2s}Step:{istep + 1:6d}{"":12s}Momentum (au)' + \
                    "".join(["\n" + f'{self.mol.symbols[iat]:5s}' + \
                    "".join([f'{self.mom[itrajectory, ist, iat, idim]:15.8f}' for idim in range(self.ndim)]) for iat in range(self.nat_qm)])
                typewriter(tmp, unixmd_dir, f"MOM_{ist}", "a")

    def print_init(self, qm, mm, restart):
        """ Routine to print the initial information of dynamics

            :param object qm: QM object containing on-the-fly calculation infomation
            :param object mm: MM object containing MM calculation infomation
            :param string restart: Option for controlling dynamics restarting
        """
        # Print initial information about molecule, qm, mm and thermostat
        super().print_init(qm, mm, False, restart)

        # Print CTMQC info.
        ct_info = textwrap.dedent(f"""\
        {"-" * 68}
        {"CTMQC Information":>43s}
        {"-" * 68}
          rho_threshold            = {self.rho_threshold:>16f}
          l_crunch                 = {self.l_crunch:>16}
          l_dc_w_mom               = {self.l_dc_w_mom:>16}
          l_traj_gaussian          = {self.l_traj_gaussian:>16}
          t_cons                   = {self.t_cons:>16d}
          l_etot0                  = {self.l_etot0:>16}
          l_lap                    = {self.l_lap:>16}
        """)

        print (ct_info, flush=True)

        # Print istate
        istate_info = textwrap.dedent(f"""\
        {"-" * 68}
        {"Initial State Information":>43s}
        {"-" * 68}
        """)
        istate_info += f"  istates (1:{self.ntrajs})             =\n"
        nlines = self.ntrajs // 6
        if (self.ntrajs % 6 == 0):
            nlines -= 1

        for iline in range(nlines + 1):
            iline1 = iline * 6
            iline2 = (iline + 1) * 6
            if (iline2 > self.ntrajs):
                iline2 = self.ntrajs
            istate_info += f"  {iline1 + 1:>4d}:{iline2:<4d};"
            istate_info += "".join([f'{str(istate):7s}' for istate in self.istates[iline1:iline2]])
            istate_info += "\n"
        print (istate_info, flush=True)

        # Print dynamics information for start line
        dynamics_step_info = textwrap.dedent(f"""\

        {"-" * 118}
        {"Start Dynamics":>65s}
        {"-" * 118}
        """)

        # Print INIT for each trajectory at each step
        INIT = f" #INFO_TRAJ{'STEP':>8s}{'Kinetic(H)':>15s}{'Potential(H)':>15s}{'Total(H)':>13s}{'Temperature(K)':>17s}{'norm':>8s}"
        dynamics_step_info += INIT

        print (dynamics_step_info, flush=True)

    def print_step(self, istep, itrajectory):
        """ Routine to print each trajectory infomation at each step about dynamics

            :param integer istep: Current MD step
            :param integer itrajectory: Current trajectory
        """
        ctemp = self.mol.ekin * 2. / float(self.mol.ndof) * au_to_K
        norm = np.sum(np.diag(self.mol.rho.real))

        # Print INFO for each step
        INFO = f" INFO_{itrajectory+1}{istep + 1:>9d}"
        INFO += f"{self.mol.ekin:14.8f}{self.mol.epot:15.8f}{self.mol.etot:15.8f}"
        INFO += f"{ctemp:13.6f}"
        INFO += f"{norm:11.5f}"
        print (INFO, flush=True)
        
        # Print event in CTMQC
        for category, events in self.event.items():
            if (len(events) != 0):
                for ievent in events:
                    print (f" {category}{istep + 1:>9d}  {ievent}", flush=True)
        self.event["DECO"] = []
