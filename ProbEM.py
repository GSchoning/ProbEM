import pandas as pd
import numpy as np
import simpeg.electromagnetics.time_domain as tdem
from simpeg.utils import plot_1d_layer_model
import discretize
import tqdm
from simpeg import maps
from simpeg.data import Data # Explicitly import Data class
from simpeg import directives
from simpeg.data_misfit import L2DataMisfit
from simpeg import regularization, inverse_problem
from simpeg import optimization
from simpeg import inversion
from simpeg import utils
from simpeg.utils import mkvc
import dask
import sys
import scipy.stats as stats
from dask.distributed import Client, progress, LocalCluster, as_completed
from discretize import TensorMesh
import os
import gstools as gs
from dask.graph_manipulation import bind
import json
from scipy.spatial.distance import pdist, squareform


cwd = r'\\winhpc\Users\GWater\schoningg\REPOS\AEM_RML'  # os.getcwd()


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


# added 2025-02-05
def is_scalar(obj):
    if isinstance(obj, np.ndarray):
        return obj.shape == ()  # Empty tuple means scalar
    else:
        return np.isscalar(obj)  # Handle other typ


# added 2025-02-05
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
        return 0.0 # Or raise an error for empty data list

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
        E = np.array([np.random.normal(0, (np.abs(isounding.dobs[i]) * (isounding.relerr)[i])) for i in range(0, len(isounding.data_object.dobs))])
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
        cdf_results = [cdf_for_value(DOIi, x) for x in np.arange(0, np.ceil(isounding.Depths.max()))]
    return cdf_results

# -----------------------

class Calibration:
    use_weights = False
    regshema = "SMOOTH"
    useREF = True

    maxIter = 20
    maxIterLS = 20
    maxIterCG = 10
    tolCG = 1e-3
    lower = 0.0001
    upper = 10

    beta0_ratio = 1e1

    max_irls_iterations = 30
    minGNiter = 1
    coolEpsFact = 1.5
    update_beta = True
    verbose = False
    Stochastic = False # Added default for Stochastic

    def __init__(self):
        pass

    def calibrate(self, Sounding, regmodel, cwd=cwd):

        sys.path.append(cwd)
        from libraries import curl

        def calfunc(self, Sounding, regmodel):
            sys.path.append(cwd)
            from libraries import curl
            from simpeg.data import Data as SimPEGData # Import simpeg.data.Data inside calfunc to ensure consistency

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
                print("no weighting_selected")

            # Define the regularization (model objective function)
            self.reg_map = maps.IdentityMap(nP=Sounding.mesh.nC)
            if self.regshema == "WLS":
                self.reg = regularization.WeightedLeastSquares(
                    Sounding.mesh,
                    mapping=self.reg_map
                )
                self.reg.reference_model = self.regmodel
            elif self.regshema == "SPARSE":
                self.reg = regularization.Sparse(Sounding.mesh, mapping=self.reg_map)
                self.reg.reference_model = self.regmodel
                self.p = 0.0
                self.q = 0.0
                self.reg.norms = [self.p, self.q]
            elif self.regshema == "SMOOTH":
                self.reg = regularization.SmoothnessFirstOrder(
                    Sounding.mesh,
                    mapping=self.reg_map,
                    orientation="x",
                    reference_model_in_smooth=self.useREF,
                )

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

            # self.opt = optimization.InexactGaussNewton(
            # maxIter=self.maxIter, maxIterLS=self.maxIterLS, maxIterCG=self.maxCG, tolCG=self.tolCG
            # )
            # Define the inverse problem
            self.inv_prob = inverse_problem.BaseInvProblem(self.dmis, self.reg, self.opt)

            # Defining a starting value for the trade-off parameter (beta) between the data
            # misfit and the regularization.
            self.starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=self.beta0_ratio)

            # Update the preconditionner
            self.update_Jacobi = directives.UpdatePreconditioner()

            # Options for outputting recovered models and predicted data for each beta.
            self.save_iteration = directives.SaveOutputEveryIteration(save_txt=False)

            # Directives for the IRLS
            self.update_IRLS = directives.UpdateIRLS( # Corrected from Update_IRLS to UpdateIRLS
                max_irls_iterations=self.max_irls_iterations,
                # minGNiter=self.minGNiter, # Removed minGNiter as it's not a recognized parameter
                # coolEpsFact=self.coolEpsFact, # Removed coolEpsFact as it's not a recognized parameter
                # update_beta=self.update_beta, # Removed update_beta as it's not a recognized parameter
            )

            # Add sensitivity weights
            self.sensitivity_weights = directives.UpdateSensitivityWeights()

            # The directives are defined as a list.
            self.directives_list = [
                self.sensitivity_weights,
                self.starting_beta,
                self.save_iteration,
                self.update_IRLS,
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
            self.rele = np.mean(np.abs((Sounding.dobs - self.pred.dclean) / Sounding.dobs))
            self.DOI = get_DOI(isounding=Sounding, Cali=self)

        if self.verbose:
            calfunc(self, Sounding, regmodel)

        else:
            with NoStdStreams():
                calfunc(self, Sounding, regmodel)

        if self.Stochastic:
            # return self.values, self.rele, self.pred
            # added 2025-02-05
            return {
                "values": self.values,
                "rele": self.rele,
                "pred": self.pred,
                "DOI": self.DOI,
            }
        else: # Added for deterministic inversion
            return {
                "values": self.values,
                "rele": self.rele,
                "pred": self.pred,
                "DOI": self.DOI,
            }


class Sounding:
    """Sounding class

    Parameters
    ----------

    Survey: AEM_preproc.Survey() class
        ABC
    iline: int
        Flight line number.
    time: int, float or str.
        Time of the sounding (within the flight line).
    inv_thickness:  float array, 1D.
        Arbitrary thicknesses to be used for the inversion.

    Attributes
    ----------

    station_data: pandas.DataFrame
        Pandas dataframe of only one row, containing the station data for the current line + time.
    UTMX:
        X coord of the sounding.
    UTMY:
        Y coord of the sounding.
    TX_ALTITUDE:
        Altitude of the transmitter.
    RX_ALTITUDE:
        Altitude of the receiver.
    station_lm_data:
        ABC
    station_hm_data:
        ABC
    station_lm_std:
        ABC
    station_hm_std:
        ABC



    Methods
    -------
    __init__(fish):
        Constructor, fish is a str which self.bar will be set to.
    Calibrate(self, regmodel):
        ABC
    def get_RML_reals(self, nreals, Lrange=50, ival=0.01, lower=0.001, upper=10, tpw=1, memlim="4GB"):
        ABC

    """

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
        self.runc_offset = Survey.Data.runc_offset

        self.station_data = Survey.Data.station_data[
            Survey.Data.station_data.index == (iline, time)
        ]
        self.UTMX = self.station_data.UTMX.values[0]
        self.UTMY = self.station_data.UTMY.values[0]
        self.Elevation = self.station_data.ELEVATION.values[0]
        self.TX_ALTITUDE = self.station_data.TX_ALTITUDE.values[0]
        self.RX_ALTITUDE = self.station_data.RX_ALTITUDE.values[0]

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

        self.tx_loc = Survey.tx_shape + [self.UTMX, self.UTMY, self.TX_ALTITUDE]
        self.rx_loc = Survey.rx_offset + [self.UTMX, self.UTMY, self.RX_ALTITUDE]
        self.tx_area = Survey.tx_area

        # Low moment souurce
        rx_lm = tdem.receivers.PointMagneticFluxTimeDerivative(
            self.rx_loc, Survey.lm_times, orientation="z"
        )

        lm_wave = tdem.sources.PiecewiseLinearWaveform(
            Survey.lm_wave_time, Survey.lm_wave_form
        )
        src_lm = tdem.sources.LineCurrent(rx_lm, self.tx_loc, waveform=lm_wave)

        # high moment source
        rx_hm = tdem.receivers.PointMagneticFluxTimeDerivative( # Corrected typo here
            self.rx_loc, Survey.hm_times, orientation="z"
        )

        hm_wave = tdem.sources.PiecewiseLinearWaveform(
            Survey.hm_wave_time, Survey.hm_wave_form
        )
        src_hm = tdem.sources.LineCurrent(rx_hm, self.tx_loc, waveform=hm_wave)

        # Survey
        self.srv = tdem.Survey([src_lm, src_hm])
        # Corrected: self.station_lm_data and self.station_hm_data already contain the actual response
        self.dobs = np.r_[self.station_lm_data, self.station_hm_data]
        self.times = np.r_[Survey.lm_times, Survey.hm_times]
        # Corrected: self.station_lm_std and self.station_hm_std already contain the actual standard deviation
        self.uncertainties = np.r_[self.station_lm_std, self.station_hm_std]

        # Use runc_offset directly as the relative error, ensuring it's positive
        self.relerr = np.ones_like(self.dobs) * self.runc_offset

        if (self.use_relerr) & (self.unc != None):
            self.relerr = np.array([self.unc for i in self.dobs])
            # If unc is provided, uncertainties are re-calculated based on absolute dobs and unc
            # Corrected: Base uncertainty on self.dobs (which is now correctly scaled) and unc
            self.uncertainties = np.abs(self.dobs) * self.unc

        # Define the data object
        if self.use_relerr:
            self.data_object = Data(
                self.srv, dobs=self.dobs, relative_error=self.relerr
            )
        else:
            self.data_object = Data(
                self.srv, dobs=self.dobs, standard_deviation=self.uncertainties
            )

        # Define a mesh for plotting and regularization.
        self.mesh = TensorMesh([(np.r_[self.inv_thickness, self.inv_thickness[-1]])], "0")
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
        self.RML.get_prior_reals_VAR(self, nreals)

        self.RML.get_perturbed_data(
            self, nreals
        )  # generate_decreasing_samples(self, nreals) #this is the decreasing samples
        self.RML.prep_parruns(self, nreals)


class RML:
    def __init__(self, Lrange, ival, lower, upper, tpw, memlim):
        self.Lrange = Lrange
        self.ival = ival
        self.lower = lower
        self.upper = upper
        self.tpw = tpw
        self.memlim = memlim

    def get_prior_reals_VAR(self, Sounding, nreals):
        self.nreals = nreals
        self.var = ((np.log10(self.lower * 10) - np.log10(self.upper / 10)) / 4) ** 2
        self.model = gs.Gaussian(
            dim=1, var=self.var, len_scale=self.Lrange, mean=np.log10(self.ival)
        )
        x = np.r_[
            Sounding.inv_thickness.cumsum(),
            [Sounding.inv_thickness.cumsum()[-1] + Sounding.inv_thickness[-1]],
        ]

        self.Depths = x

        self.srf = gs.SRF(self.model, mean=np.log10(self.ival))
        self.srf(x, mesh_type="unstructured")
        self.seeds = np.random.randint(low=1, high=11111111, size=self.nreals)
        self.fields = []
        for seed in self.seeds:
            seed = gs.random.MasterRNG(seed)
            field = self.srf(seed=seed())
            self.fields.append(10**field)

    def get_perturbed_data(self, Sounding, nreals):
        pobs = []
        for index in range(len(Sounding.dobs)):
            obs = Sounding.dobs[index]
            std = np.abs(obs) * Sounding.relerr[index]  # Sounding.uncertainties[index]
            obsreals = np.random.normal(obs, std, nreals)
            obsreals = np.abs(obsreals) * -1

            pobs.append(obsreals)
        self.pobs = np.array(pobs).T
        return pobs

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
                lower_bound = stats.norm.ppf(0.0001, loc=observations[j], scale=std_devs[j])

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
            Cbi.regshema = "SMOOTH"
            Cbi.Stochastic = True
            real = self.fields[i]
            pert_obs = self.pobs[i]
            Sounding.data_object = Data(
                Sounding.srv, dobs=pert_obs, relative_error=Sounding.relerr
            )
            lazy_result = dask.delayed(Cbi.calibrate)(Sounding, real)
            self.lazy_results.append(lazy_result)

    def run_local(self, cluster=None, client=None):
        self.ncores = int(os.cpu_count()) - 1

        if (cluster is None) and (client is None):
            self.closeflag = True
            cluster = LocalCluster(
                threads_per_worker=self.tpw, n_workers=self.ncores, memory_limit=self.memlim
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

        self.p50 = np.quantile(self.calreals, 0.5, axis=0)
        self.p5 = np.quantile(self.calreals, 0.05, axis=0)
        self.p95 = np.quantile(self.calreals, 0.95, axis=0)
        self.pprob = np.sum(peaks, axis=0) / len(peaks)
        self.tprob = np.sum(troughs, axis=0) / len(peaks)

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
            self.count = np.zeros_like(self.cdf) # Placeholder


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
