import json
import os
import sys

import dask
import gstools as gs
import numpy as np
import pandas as pd
import scipy.stats as stats
import simpeg.electromagnetics.time_domain as tdem
from dask.distributed import Client, LocalCluster
from discretize import TensorMesh
from scipy.signal import find_peaks
from simpeg import (
    directives,
    inverse_problem,
    inversion,
    maps,
    optimization,
    regularization,
)
from simpeg.data import Data  # Explicitly import Data class
from simpeg.data_misfit import L2DataMisfit

cwd = r"\\winhpc\Users\GWater\schoningg\REPOS\AEM_RML"  # os.getcwd()


class NoStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self.devnull = open(os.devnull, "w")
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


def fsim(srv, mesh, pars):
    model_mapping = maps.ExpMap(nP=mesh.nC)
    simulation = tdem.Simulation1DLayered(
        survey=srv, thicknesses=mesh.h[0][:-1], sigmaMap=model_mapping
    )

    pred = simulation.make_synthetic_data(np.log(pars))

    reg_map = maps.IdentityMap(nP=mesh.nC)
    reg = regularization.WeightedLeastSquares(mesh, mapping=reg_map)
    reg = reg(pars)

    return pred, reg


def is_scalar(obj):
    if isinstance(obj, np.ndarray):
        return obj.shape == ()  # Empty tuple means scalar
    else:
        return np.isscalar(obj)  # Handle other typ


def get_noise_real(dobs, noise=0.03):
    # 1. Convert dobs_log to normal space
    if is_scalar(noise):
        # 2. Calculate the standard deviation for each value in normal space
        noise = np.abs(dobs) * np.abs(noise)

    reals = np.abs(np.random.normal(dobs, np.abs(noise), size=len(dobs))) * -1
    noise_log = dobs - reals
    return noise_log


# --- New DOI Methods ---


def get_cutoff(isounding, S, V, kmin=0.0001, kmax=10):
    kmin, kmax = 0.0001, 10  # bounds in S/m
    S2k = ((np.log(kmax) - np.log(kmin)) / 4) ** 2
    S2k = ((kmax - kmin) / 4) ** 2
    # prior parameter variance                                             # create empty y matrix
    S1inv = np.linalg.inv(np.diag(S))
    S1inv_2 = S1inv**2
    Yemp = np.zeros(np.shape(V)[0])
    svmin = []
    kt = []
    for s in range(0, np.shape(V)[1]):
        Y = Yemp.copy()
        Y[s] = 1
        P1 = []
        P2 = []
        Perrc = []
        for w in range(0, len(S)):
            S2E = (isounding.uncertainties**2)[w]
            YtV_2 = []
            for i2 in range(w + 1, np.shape(V)[1]):
                Vi = V[:, i2]

                YtV_2.append((Y.T @ Vi) ** 2)

            P1i = np.sum(YtV_2) * S2k
            P1.append(P1i)

            SiyTvi = []
            for i3 in range(1, w):
                Vi = V[:, i3]
                S2inv2YTVi = S1inv_2[i3 - 1, i3 - 1] * (Y.T @ Vi) ** 2
                SiyTvi.append(S2inv2YTVi)

            P2i = np.sum(SiyTvi) * S2E

            P2.append(P2i)
            Perrc.append(P1i + P2i)
            k = np.argmin(Perrc)
            kt.append(k)
    return int(np.mean(kt))


def cdf_for_value(data, x):
    """
    Calculates the empirical CDF for a given value x.

    Args:
      data (list or np.ndarray): A list or array of numerical data.
      x (float or int): The value at which to evaluate the CDF.

    Returns:
      float: The CDF value, which is the proportion of data <= x.
    """
    # Ensure data is a numpy array for efficient comparison
    data_array = np.array(data)

    # Count how many items in data_array are less than or equal to x
    count = np.sum(data_array <= x)

    # Divide by the total number of items to get the proportion (the CDF value)
    total_count = len(data_array)

    if total_count == 0:
        return 0.0  # Or raise an error for empty data list

    return count / total_count


def get_DOI(isounding, Cali, depths=False):
    # Replaced ["ds"] with direct access as getJ returns array in this env
    JW = isounding.simulation.getJ(m=np.log(Cali.values))

    U, S, VT = np.linalg.svd(JW)
    V = VT.T
    k = get_cutoff(isounding, S, V, kmin=0.0001, kmax=3)

    V1 = V[:, :k]
    V2 = V[:, k:]
    R = V1 @ V1.T

    S1 = np.diag(S[:k])
    S1inv = np.linalg.inv(S1)
    U1 = U[:, :k]
    DOIlist = []
    LOIlist = []
    NR = []

    rats = []
    DOIi = []
    nreals = 10000
    for i in range(0, nreals):
        rat = []
        E = np.array(
            [
                np.random.normal(0, (np.abs(isounding.dobs[i]) * (isounding.relerr)[i]))
                for i in range(0, len(isounding.data_object.dobs))
            ]
        )
        # E=np.log10(E)
        noise_response = V1 @ S1inv @ U1.T @ E
        for r in range(len(R)):
            kval = np.log(Cali.values)[r]
            real = R[r]
            # maxTrue = np.max(np.abs(real*kval))
            maxNoise = np.max(abs(noise_response))
            maxTrue = np.max(np.abs(real[r] * kval))
            maxNoise = np.max(abs(noise_response[r]))
            nR = maxTrue / maxNoise
            rat.append(nR)
        rats.append(rat)
        DOIi.append(isounding.Depths[np.cumsum(np.array(rat) > 1).argmax()])
    if depths == False:
        cdf_results = [cdf_for_value(DOIi, x) for x in isounding.Depths]
    else:
        cdf_results = [
            cdf_for_value(DOIi, x)
            for x in np.arange(0, np.ceil(isounding.Depths.max()))
        ]
    return cdf_results


# -----------------------


class Calibration:
    use_weights = False
    regshema = "WLS"
    useREF = True

    maxIter = 30
    maxIterLS = 20
    maxIterCG = 10
    tolCG = 1e-3
    lower = 0.0001
    upper = 10

    beta0_ratio = 1e1

    max_irls_iterations = 50
    minGNiter = 1
    coolEpsFact = 2
    update_beta = True
    verbose = False
    Stochastic = False  # Added default for Stochastic

    def __init__(self):
        pass

    def calibrate(self, Sounding, regmodel, cwd=cwd):
        sys.path.append(cwd)
        # Assuming libraries import handles recursive imports if needed
        # from libraries import curl

        def calfunc(self, Sounding, regmodel):
            # sys.path.append(cwd)
            # from libraries import curl
            # from simpeg.data import Data as SimPEGData

            self.regmodel = np.log(regmodel)
            self.inv_thickness = Sounding.inv_thickness
            # Define mapping from model to active cells.

            # Define the data misfit. Here the data misfit is the L2 norm of the weighted
            # residual between the observed data and the data predicted for a given model.
            # The weighting is defined by the reciprocal of the uncertainties.

            self.dmis = L2DataMisfit(
                simulation=Sounding.simulation, data=Sounding.data_object
            )

            if self.use_weights:
                self.dmis.W = 1.0 / Sounding.uncertainties
            else:
                # print("no weighting_selected")
                pass

            # Define the regularization (model objective function)
            self.reg_map = maps.IdentityMap(nP=Sounding.mesh.nC)
            if self.regshema == "WLS":
                self.reg = regularization.WeightedLeastSquares(
                    Sounding.mesh, mapping=self.reg_map, alpha_s=1.0, alpha_x=10
                )
                wx = np.ones(Sounding.mesh.nC)
                wx[0:3] = 5.0
                self.reg.objfcts[1].set_weights(surface_lock=wx)

            elif self.regshema == "SPARSE":
                self.reg = regularization.Sparse(Sounding.mesh, mapping=self.reg_map)
                self.reg.reference_model = self.regmodel
                self.p = 2
                self.q = 1
                self.reg.norms = [self.p, self.q]
            elif self.regshema == "SMOOTH":
                self.reg = regularization.SmoothnessFirstOrder(
                    Sounding.mesh,
                    mapping=self.reg_map,
                    orientation="x",
                )
            self.reg.reference_model = self.regmodel
            self.reg.reference_model_in_smooth = self.useREF

            # Define how the optimization problem is solved. Here we will use an inexact
            # Gauss-Newton approach that employs the conjugate gradient solver.

            self.opt = optimization.ProjectedGNCG(
                maxIter=self.maxIter,
                maxIterLS=self.maxIterLS,
                maxIterCG=self.maxIterCG,
                tolCG=self.tolCG,
                lower=np.log(self.lower),
                upper=np.log(self.upper),
            )

            # Define the inverse problem
            self.inv_prob = inverse_problem.BaseInvProblem(
                self.dmis, self.reg, self.opt
            )

            # Defining a starting value for the trade-off parameter (beta) between the data
            # misfit and the regularization.
            self.starting_beta = directives.BetaEstimate_ByEig(
                beta0_ratio=self.beta0_ratio
            )

            # Update the preconditionner
            self.update_Jacobi = directives.UpdatePreconditioner()

            # Options for outputting recovered models and predicted data for each beta.
            self.save_iteration = directives.SaveOutputEveryIteration(save_txt=False)

            # Directives for the IRLS
            self.update_IRLS = directives.UpdateIRLS(
                max_irls_iterations=self.max_irls_iterations,
            )

            # Add sensitivity weights
            self.sensitivity_weights = directives.UpdateSensitivityWeights()

            # 2. Cool Beta (Reduce regularization over time)
            self.beta_schedule = directives.BetaSchedule(
                coolingFactor=self.coolEpsFact, coolingRate=1
            )  # Cool every iteration)

            self.target_misfit = directives.TargetMisfit(chifact=1.0)

            # The directives are defined as a list.
            if self.regshema == "SPARSE":
                self.directives_list = [
                    self.sensitivity_weights,
                    self.starting_beta,
                    self.save_iteration,
                    self.update_IRLS,
                    self.update_Jacobi,
                ]
            else:
                self.directives_list = [
                    self.sensitivity_weights,
                    self.starting_beta,
                    self.beta_schedule,  # Replaces IRLS cooling
                    self.target_misfit,  # Stops the inversion
                    self.save_iteration,
                    self.update_Jacobi,
                ]

            self.inv = inversion.BaseInversion(self.inv_prob, self.directives_list)

            # Run the inversion

            self.recovered_model = self.inv.run(self.regmodel)

            self.values = Sounding.model_mapping * self.recovered_model

            pred, reg = fsim(Sounding.srv, Sounding.mesh, self.values)
            self.pred = pred
            self.CHi2 = np.sqrt(
                1
                / len(Sounding.dobs)
                * np.sum(
                    (Sounding.dobs - self.pred.dclean) ** 2
                    / (Sounding.tx_area * Sounding.uncertainties) ** 2
                )
            )
            self.rele = np.mean(
                np.abs((Sounding.dobs - self.pred.dclean) / Sounding.dobs)
            )
            self.DOI = get_DOI(isounding=Sounding, Cali=self)

        if self.verbose:
            calfunc(self, Sounding, regmodel)

        else:
            with NoStdStreams():
                calfunc(self, Sounding, regmodel)

        return {
            "values": self.values,
            "rele": self.rele,
            "pred": self.pred,
            "DOI": self.DOI,
        }


class Sounding:
    """Sounding class"""

    def __init__(self, Survey, iline, time, inv_thickness, use_relerr=True, unc=None):
        self.iline = iline
        self.time = time
        self.inv_thickness = inv_thickness
        self.Depths = np.r_[
            self.inv_thickness.cumsum(),
            self.inv_thickness.cumsum()[-1] + self.inv_thickness[-1],
        ]
        self.use_relerr = use_relerr
        self.unc = unc

        # Pull global settings from Survey.Data if available
        # Default to 0.03 if not found to prevent crashes
        try:
            self.runc_offset = Survey.Data.runc_offset
        except AttributeError:
            self.runc_offset = 0.03

        # ------------------------------------------------------------------
        # 1. LOAD STATION DATA
        # ------------------------------------------------------------------
        self.station_data = Survey.Data.station_data[
            Survey.Data.station_data.index == (iline, time)
        ]
        self.UTMX = self.station_data.UTMX.values[0]
        self.UTMY = self.station_data.UTMY.values[0]
        self.Elevation = self.station_data.ELEVATION.values[0]
        self.TX_ALTITUDE = self.station_data.TX_ALTITUDE.values[0]
        self.RX_ALTITUDE = self.station_data.RX_ALTITUDE.values[0]

        # Load Data Arrays (Already trimmed by AEM_preproc)
        self.station_lm_data = Survey.Data.lm_data[
            Survey.Data.lm_data.index == (iline, time)
        ].to_numpy(dtype=float)[0]
        self.station_hm_data = Survey.Data.hm_data[
            Survey.Data.hm_data.index == (iline, time)
        ].to_numpy(dtype=float)[0]

        self.station_lm_std = Survey.Data.lm_std[
            Survey.Data.lm_std.index == (iline, time)
        ].to_numpy(dtype=float)[0]
        self.station_hm_std = Survey.Data.hm_std[
            Survey.Data.hm_std.index == (iline, time)
        ].to_numpy(dtype=float)[0]

        # ------------------------------------------------------------------
        # 2. DEFINE SYSTEM GEOMETRY
        # ------------------------------------------------------------------
        self.tx_loc = Survey.tx_shape + [self.UTMX, self.UTMY, self.TX_ALTITUDE]
        self.rx_loc = Survey.rx_offset + [self.UTMX, self.UTMY, self.RX_ALTITUDE]
        self.tx_area = Survey.tx_area

        # Low moment source
        rx_lm = tdem.receivers.PointMagneticFluxTimeDerivative(
            self.rx_loc, Survey.lm_times, orientation="z"
        )
        lm_wave = tdem.sources.PiecewiseLinearWaveform(
            Survey.lm_wave_time, Survey.lm_wave_form
        )
        src_lm = tdem.sources.LineCurrent(rx_lm, self.tx_loc, waveform=lm_wave)

        # High moment source
        rx_hm = tdem.receivers.PointMagneticFluxTimeDerivative(
            self.rx_loc, Survey.hm_times, orientation="z"
        )
        hm_wave = tdem.sources.PiecewiseLinearWaveform(
            Survey.hm_wave_time, Survey.hm_wave_form
        )
        src_hm = tdem.sources.LineCurrent(rx_hm, self.tx_loc, waveform=hm_wave)

        # ------------------------------------------------------------------
        # 3. CONSOLIDATE DATA & CALCULATE ROBUST UNCERTAINTIES
        # ------------------------------------------------------------------
        self.srv = tdem.Survey([src_lm, src_hm])
        self.dobs = np.r_[self.station_lm_data, self.station_hm_data]
        self.times = np.r_[Survey.lm_times, Survey.hm_times]

        # A. Start with Relative Error + Noise Floor
        #    Note: 1e-15 is good for synthetics. Consider 1e-14 for field data.
        noise_floor = 1e-15

        # If user supplied a specific uncertainty override (unc), use it.
        # Otherwise use the data-derived relative error + floor.
        if (self.use_relerr) & (self.unc is not None):
            self.relerr = np.array([self.unc for i in self.dobs])
            self.uncertainties = np.abs(self.dobs) * self.unc
        else:
            self.relerr = np.ones_like(self.dobs) * self.runc_offset
            self.uncertainties = np.sqrt(
                (self.dobs * self.runc_offset) ** 2 + noise_floor**2
            )

        # B. APPLY SKYTEM RAMP SAFETY FACTOR (The Artifact Fix)
        #    De-weight the first 4 VALID gates of the Low Moment to suppress ramp artifacts.
        #    Since pre-proc removes garbage gates, index 0 is actually Gate 8 (First valid).
        safety_factor = np.ones_like(self.uncertainties)

        # Apply 3x penalty to the first 4 gates
        safety_factor[0:1] = 3.0

        # Multiply uncertainties by this safety mask
        self.uncertainties = self.uncertainties * safety_factor

        # ------------------------------------------------------------------
        # 4. INITIALIZE SIMPEG DATA OBJECT (CRITICAL FIX)
        # ------------------------------------------------------------------
        # We MUST use standard_deviation=self.uncertainties.
        # If we used relative_error=..., SimPEG would ignore our floor and safety factor.

        self.data_object = Data(
            self.srv, dobs=self.dobs, standard_deviation=self.uncertainties
        )

        # ------------------------------------------------------------------
        # 5. INITIALIZE MESH & SIMULATION
        # ------------------------------------------------------------------
        self.mesh = TensorMesh(
            [(np.r_[self.inv_thickness, self.inv_thickness[-1]])], "0"
        )
        self.model_mapping = maps.ExpMap(nP=self.mesh.nC)

        self.simulation = tdem.Simulation1DLayered(
            survey=self.srv, thicknesses=self.inv_thickness, sigmaMap=self.model_mapping
        )

    def Calibrate(self, regmodel):
        self.Calibration = Calibration()
        self.Calibration.calibrate(self, regmodel)

    def get_RML_reals(
        self, nreals, Lrange=20, ival=0.05, lower=0.0001, upper=10, tpw=1, memlim="4GB"
    ):
        self.RML = RML(
            Lrange=Lrange, ival=ival, lower=lower, upper=upper, tpw=tpw, memlim=memlim
        )
        self.RML.get_prior_reals_CONV(self, nreals)

        self.RML.get_perturbed_data(self, nreals)
        self.RML.prep_parruns(self, nreals)


class RML:
    def __init__(self, Lrange, ival, lower, upper, tpw, memlim):
        self.Lrange = Lrange
        self.ival = ival
        self.lower = lower
        self.upper = upper
        self.tpw = tpw
        self.memlim = memlim

    def get_prior_reals_CONV(self, Sounding, nreals):
        """
        Generates 'Background + Anomalies' priors using Matrix Convolution.
        Produces peaks/troughs oscillating around a fixed background (ival).
        """
        # 1. Grab Depths
        self.Depths = Sounding.Depths
        self.nreals = nreals

        # 2. Setup Geometry (Layer Midpoints)
        z_bot = Sounding.inv_thickness.cumsum()
        z_top = np.r_[0, z_bot[:-1]]
        z_mid = (z_top + z_bot) / 2.0

        # Fix dimensions for SimPEG (30 cells)
        n_cells = Sounding.mesh.nC
        if len(z_mid) == n_cells - 1:
            z_mid = np.r_[z_mid, z_bot[-1] + 50.0]
        if len(z_mid) != n_cells:
            z_mid = np.linspace(0, z_bot[-1] + 100, n_cells)

        # --- 3. DEFINE FEATURE WIDTH (Correlation Length) ---
        # To avoid trends, these numbers must be significantly smaller than total depth.
        # 10m at surface -> 40m at depth creates distinct "layers" rather than a drift.
        L_surface = 30.0
        L_bottom = 80.0

        length_scales = np.interp(z_mid, [0, 250], [L_surface, L_bottom])

        # --- 4. BUILD CORRELATION MATRIX (Exponential Kernel) ---
        # Exponential (blockier) vs Gaussian (smoother).
        # Exponential is often better for "peaks and troughs".
        nC = len(z_mid)
        C = np.eye(nC)

        for i in range(nC):
            for j in range(nC):
                if i == j:
                    continue
                dist = abs(z_mid[i] - z_mid[j])
                L_local = (length_scales[i] + length_scales[j]) / 2.0

                # Exponential Kernel: exp( - distance / length )
                # Produces "rougher" layers (Markov process)
                C[i, j] = np.exp(-1.0 * (dist / L_local))

        # --- 5. CHOLESKY DECOMPOSITION ---
        # Add jitter for stability
        C += np.eye(nC) * 1e-6
        try:
            L_mat = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            U, S, Vt = np.linalg.svd(C)
            L_mat = U @ np.diag(np.sqrt(S))

        # --- 6. GENERATE ---
        # Calculate Log-Space Statistics
        log_lower = np.log10(self.lower)
        log_upper = np.log10(self.upper)
        var_log = ((log_upper - log_lower) / 6.0) ** 2
        std_log = np.sqrt(var_log)

        # Center the mean exactly on your background (ival)
        # We assume the output of the convolution is mean=0, so we just add log(ival)
        log_background = np.log10(self.ival)

        self.fields = []
        MIN_LOG_COND = -4.0
        MAX_LOG_COND = 0.5

        for _ in range(self.nreals):
            # Generate Fluctuations
            white_noise = np.random.randn(nC)
            fluctuations = L_mat @ white_noise

            # Combine: Background + (Fluctuation * Magnitude)
            log_model = log_background + (fluctuations * std_log)

            # Clip and Convert to Linear S/m
            self.fields.append(10 ** (np.clip(log_model, MIN_LOG_COND, MAX_LOG_COND)))

        self.prior_matrix = np.array(self.fields)

    def get_prior_reals_VAR(self, Sounding, nreals):
        """
        Generates 1D non-stationary prior realizations using gstools.
        Automatically handles the 'Infinite Basement' dimension mismatch.
        """
        # --- CRITICAL FIX: Grab Depths from Sounding ---
        self.Depths = Sounding.Depths
        # -----------------------------------------------

        self.nreals = nreals

        # --- 1. SETUP VARIANCE & MODEL ---
        log_lower = np.log10(self.lower)
        log_upper = np.log10(self.upper)
        self.var = ((log_upper - log_lower) / 4.0) ** 2

        # Mean correction for Log-Normal bias
        correction_factor = 0.5 * self.var * np.log(10)
        corrected_log_mean = np.log10(self.ival) - correction_factor

        # Log-Space Length Scale (0.3 ~ doubling of thickness correlation)
        log_Lrange = 0.15
        self.model = gs.Gaussian(dim=1, var=self.var, len_scale=log_Lrange)

        # --- 2. FIX COORDINATES ---
        # Target: We need exactly Sounding.mesh.nC points (e.g., 30)
        n_cells = Sounding.mesh.nC

        # Calculate finite layer bottoms
        z_bot = Sounding.inv_thickness.cumsum()  # shape (29,)
        z_top = np.r_[0, z_bot[:-1]]  # shape (29,)
        z_mid = (z_top + z_bot) / 2.0  # shape (29,)

        # Check if we are missing the basement point
        if len(z_mid) == n_cells - 1:
            # Create a midpoint for the 30th (infinite) layer
            # We just place it 50m below the last finite layer
            basement_depth = z_bot[-1] + 50.0
            z_mid = np.r_[z_mid, basement_depth]

        # Ensure shapes match now
        if len(z_mid) != n_cells:
            # Fallback for weird meshes: just linspace it
            print(
                f"Warning: Mesh cells ({n_cells}) != Depth points ({len(z_mid)}). Resizing..."
            )
            z_mid = np.linspace(0, z_bot[-1] + 100, n_cells)

        # Apply Log-Depth Scaling
        shift_factor = 60.0
        z_log_coords = np.log10(z_mid + shift_factor)

        # --- 3. GENERATE ---
        self.srf = gs.SRF(self.model, mean=corrected_log_mean)
        self.seeds = np.random.randint(low=1, high=11111111, size=self.nreals)
        self.fields = []

        # Safety Limits (Log10 S/m)
        MIN_LOG_COND = -4.0
        MAX_LOG_COND = 0.5

        for seed in self.seeds:
            seed_rng = gs.random.MasterRNG(seed)

            # Generate exactly n_cells points
            raw_log_field = self.srf(
                z_log_coords, mesh_type="unstructured", seed=seed_rng()
            )

            # Clip and Save
            self.fields.append(
                10 ** (np.clip(raw_log_field, MIN_LOG_COND, MAX_LOG_COND))
            )

        # Shape should now be (n_reals, 30) -> Fits SimPEG perfectly
        self.prior_matrix = np.array(self.fields)

    def get_perturbed_data(self, Sounding, nreals):
        pobs = []
        for index in range(len(Sounding.dobs)):
            obs = Sounding.dobs[index]

            # OLD: std = np.abs(obs) * Sounding.relerr[index] (Ignores floor/safety)

            # NEW: Use the robust uncertainty from the Sounding
            std = Sounding.uncertainties[index]

            # Perturb
            obsreals = np.random.normal(obs, std, nreals)

            # Ensure negative sign (for AEM magnitude) if needed,
            # though usually it's better to perturb the raw value.
            # If your obs are negative, this is fine:
            # obsreals = -1 * np.abs(obsreals) # Only if you need to enforce sign

            pobs.append(obsreals)

        self.pobs = np.array(pobs).T
        return self.pobs

    def generate_decreasing_samples(self, Sounding, nreals):
        """Generates random samples with a decreasing trend, handling edge cases."""
        observations = np.abs(Sounding.dobs)
        std_devs = np.abs(observations) * Sounding.relerr
        num_time_points = len(observations)

        if len(observations) != len(std_devs) or num_time_points == 0:
            return np.array([])

        samples = np.zeros((nreals, num_time_points))

        for i in range(nreals):
            # First point (no constraint)
            samples[i, 0] = stats.norm.rvs(loc=observations[0], scale=std_devs[0])

            for j in range(1, num_time_points):
                upper_bound = samples[i, j - 1]

                # Robust lower bound calculation (avoiding potential domain errors)
                lower_bound = stats.norm.ppf(
                    0.0001, loc=observations[j], scale=std_devs[j]
                )

                # Ensure lower_bound is strictly less than upper_bound
                lower_bound = min(lower_bound, upper_bound - 1e-9)  # A small tolerance
                if j == num_time_points:
                    lower_bound = 0
                a = (lower_bound - observations[j]) / std_devs[j]
                b = (upper_bound - observations[j]) / std_devs[j]

                samples[i, j] = stats.truncnorm.rvs(
                    a=a, b=b, loc=observations[j], scale=std_devs[j]
                )
        self.pobs = -1 * np.abs(samples)
        return -1 * np.abs(samples)

    def prep_parruns(self, Sounding, nreals):
        self.lazy_results = []
        for i in range(nreals):
            Cbi = Calibration()
            Cbi.lower = self.lower
            Cbi.upper = self.upper
            Cbi.regshema = "WLS"
            Cbi.Stochastic = True
            real = self.fields[i]
            pert_obs = self.pobs[i]
            Sounding.data_object = Data(
                Sounding.srv, dobs=pert_obs, relative_error=Sounding.relerr
            )
            lazy_result = dask.delayed(Cbi.calibrate)(Sounding, real)
            self.lazy_results.append(lazy_result)

    def calc_feature_probs(self):
        nreals = len(self.calreals)
        n_depths = len(self.Depths)

        peak_counts = np.zeros(n_depths)
        trough_counts = np.zeros(n_depths)

        # PROMINENCE: 0.1 log-units (approx 25% contrast)
        # WIDTH: 2 cells (ignores single-point spikes)
        MIN_PROMINENCE = 0.05
        MIN_WIDTH = 2

        for real in self.calreals:
            # Convert to Log10 for consistent scale
            log_real = np.log10(real + 1e-10)  # epsilon to avoid log(0)

            # Find Peaks (Conductive Layers)
            idx_peaks, _ = find_peaks(
                log_real, prominence=MIN_PROMINENCE, width=MIN_WIDTH
            )
            peak_counts[idx_peaks] += 1

            # Find Troughs (Resistive Layers) -> Peaks of the negative signal
            idx_troughs, _ = find_peaks(
                -log_real, prominence=MIN_PROMINENCE, width=MIN_WIDTH
            )
            trough_counts[idx_troughs] += 1

        self.pprob = peak_counts / nreals
        self.trough_prob = trough_counts / nreals

    def run_local(self, cluster=None, client=None):
        self.ncores = int(os.cpu_count()) - 1

        if (cluster is None) and (client is None):
            self.closeflag = True
            cluster = LocalCluster(
                threads_per_worker=self.tpw,
                n_workers=self.ncores,
                memory_limit=self.memlim,
            )
        else:
            self.closeflag = False

        if client is not None:
            self.dashboard_link = client.dashboard_link
            print(self.dashboard_link)
            sys.stdout.flush()
            results = dask.compute(*self.lazy_results)

        else:
            with Client(cluster) as client:
                self.dashboard_link = client.dashboard_link

                print(self.dashboard_link)
                sys.stdout.flush()
                results = dask.compute(*self.lazy_results)

            if self.closeflag:
                cluster.close()

        self.calreals = [x["values"] for x in results]
        self.fits = [x["rele"] for x in results]
        self.preds = [x["pred"] for x in results]
        # added 2025-02-05
        self.DOIs = [x["DOI"] for x in results]
        peaks = [
            np.r_[
                [False],
                [False],
                [
                    (real[x - 1] < real[x] > real[x + 1])
                    & (real[x - 2] < real[x] > real[x + 2])
                    for x in range(2, len(real) - 2)
                ],
                [False],
                [False],
            ]
            for real in self.calreals
        ]

        troughs = [
            np.r_[
                [False],
                [False],
                [
                    (real[x - 1] > real[x] < real[x + 1])
                    & (real[x - 2] > real[x] < real[x + 2])
                    for x in range(2, len(real) - 2)
                ],
                [False],
                [False],
            ]
            for real in self.calreals
        ]

        log_reals = np.log10(np.array(self.calreals) + 1e-12)

        # 2. Calculate Percentiles in Log Space
        log_p50 = np.quantile(log_reals, 0.5, axis=0)
        log_p5 = np.quantile(log_reals, 0.05, axis=0)
        log_p95 = np.quantile(log_reals, 0.95, axis=0)

        # 3. Convert back to Linear Space (S/m)
        self.p50 = 10**log_p50
        self.p5 = 10**log_p5
        self.p95 = 10**log_p95
        # self.pprob = np.sum(peaks, axis=0) / len(peaks)
        # self.tprob = np.sum(troughs, axis=0) / len(peaks)

        # --- NEW: Gradient Probability Logic ---
        nreals = len(self.calreals)

        self.calc_feature_probs()

        # Now calculating indices is safe because pprob exists
        dif_1 = [np.gradient(r, self.Depths) for r in self.calreals]
        self.ri_prob = np.sum([d > 0 for d in dif_1], axis=0) / self.nreals
        self.fa_prob = np.sum([d < 0 for d in dif_1], axis=0) / self.nreals

        # Recalculate your Master Index with the new robust peaks
        self.layer_index = self.ri_prob + self.pprob - self.fa_prob
        self.polarity_index = self.ri_prob - self.fa_prob
        # Combined Layer Probability (Directional Contrast)
        self.layprob = np.where(
            self.ri_prob > self.fa_prob, self.ri_prob, self.fa_prob * -1
        )

        try:
            # Handle scalar DOIs
            self.DOI_mean = np.mean(self.DOIs)
            self.DOI_std = np.std(self.DOIs)
            bins = self.Depths
            Z = self.DOIs
            self.count, bins_count = np.histogram(Z, bins=bins)
            self.pdf = self.count / sum(self.count)
            self.cdf = np.cumsum(self.pdf)
        except:
            # Handle array DOIs (e.g. CDFs)
            # We average the CDFs
            self.DOI_mean = np.mean(self.DOIs, axis=0)
            self.DOI_std = np.std(self.DOIs, axis=0)
            self.cdf = self.DOI_mean
            self.pdf = np.diff(self.cdf, prepend=0)
            self.count = np.zeros_like(self.cdf)  # Placeholder


def adjust_dtype(var):
    if isinstance(var, np.integer):
        return int(var)
    else:
        return var


def proc_output(out, fd_output_sounding):
    fi_out_rml_tpl = r"{}\rml.gz.parquet"
    fi_out_obs_tpl = r"{}\obs.gz.parquet"
    fi_out_preds_tpl = r"{}\preds.gz.parquet"
    fi_out_vars_tpl = r"{}\variables.json"

    time, isounding = out

    df_rml = pd.DataFrame(
        {
            "depth": isounding.inv_thickness.cumsum(),
            "p5": isounding.RML.p5[:-1],
            "p50": isounding.RML.p50[:-1],
            "p95": isounding.RML.p95[:-1],
            "pprob": isounding.RML.pprob[:-1],
            "tprob": isounding.RML.tprob[:-1],
            # --- Added New Probability Columns ---
            "ri_prob": isounding.RML.ri_prob[:-1],
            "fa_prob": isounding.RML.fa_prob[:-1],
            "layprob": isounding.RML.layprob[:-1],
            # -------------------------------------
            "doicdf": isounding.RML.cdf,
        }
    )

    edict = {}
    edict["times"] = isounding.times
    edict["obs"] = isounding.dobs
    for i in range(0, len(isounding.RML.preds)):
        edict["real_" + str(i)] = isounding.RML.preds[i].dclean
    df_obs = pd.DataFrame(edict)

    caldict = {}
    caldict["depths"] = isounding.Depths

    for i in range(0, len(isounding.RML.calreals)):
        edict["real_" + str(i)] = isounding.RML.calreals[i]
    df_calreals = pd.DataFrame(caldict)
    df_calreals = pd.DataFrame(isounding.RML.calreals)

    dic_vars = {
        "lineno": adjust_dtype(isounding.iline),
        "time": adjust_dtype(isounding.time),
        "easting": adjust_dtype(isounding.UTMX),
        "northing": adjust_dtype(isounding.UTMY),
        "elevation": adjust_dtype(isounding.Elevation),
        "mean_relerr": adjust_dtype(isounding.RML.fits),
        "DOI_mean": adjust_dtype(isounding.RML.DOI_mean),
        "DOI_std": adjust_dtype(isounding.RML.DOI_std),
    }

    # create the output folder for each run (line+time)
    if not os.path.exists(fd_output_sounding):
        os.makedirs(fd_output_sounding, exist_ok=True)

    fi_out_rml = fi_out_rml_tpl.format(fd_output_sounding)
    fi_out_obs = fi_out_obs_tpl.format(fd_output_sounding)
    fi_out_vars = fi_out_vars_tpl.format(fd_output_sounding)
    fi_out_preds = fi_out_preds_tpl.format(fd_output_sounding)

    df_rml.to_parquet(fi_out_rml, index=False)
    df_obs.to_parquet(fi_out_obs, index=False)
    df_calreals.to_parquet(fi_out_preds, index=False)
    with open(fi_out_vars, "w") as f:
        json.dump(dic_vars, f, indent=4)
