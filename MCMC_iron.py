import numpy as np
from setup import *
import emcee
import multiprocessing as mp
from multiprocessing import cpu_count
import math
import h5py

# Definitions:
def log_posterior(theta, times):
    lpri = log_prior(theta)
    if not np.isfinite(lpri):
        return -np.inf
    return lpri + log_likelihood(theta, times)

# def log_prior(theta):
#     # prior without init conditions
#     # no matrix weight
#     sigma2, r_mr, r_rl, r_d, c_max = theta
#     # everything that is defined in the conditions after the first one corresponds to uniform priors
#     if all(c >= 0 for c in theta) and (0 < c_max < 1000) and (r_mr < 100):
#         a_r1, b_r1 = 10, .06
#         a_r2, b_r2 = 5, .01
#         a_r3, b_r3 = 10, .05
#         a_sig, b_sig = 10, .05
#         p_uptake_r = sp.stats.gamma.pdf(r_mr, a=a_r1, loc=0, scale=b_r1)
#         p_decay_r = sp.stats.gamma.pdf(r_d, a=a_r2, loc=0, scale=b_r2)
#         p_r2l_r = sp.stats.gamma.pdf(r_rl, a=a_r3, loc=0, scale=b_r3)
#         p_sigma2 = sp.stats.gamma.pdf(sigma2, a=a_sig, loc=0, scale=b_sig)
#         lprior = np.log(p_uptake_r*p_decay_r*p_r2l_r*p_sigma2)
#         return lprior
#     # returns a scalart
#     return -np.inf

def log_prior(theta):
    # no matrix weight
    sigma2, r_mr, r_rl, *iron_0 = theta
    # everything that is defined in the conditions after the first one corresponds to uniform priors
    if all(c >= 0 for c in theta):
        a_r1, b_r1 = 5, .01
        a_r2, b_r2 = 5, .001
        a_r3, b_r3 = 5, .01
        a_sig, b_sig = 10, .05
        p_uptake_r = sp.stats.gamma.pdf(r_mr, a=a_r1, loc=0, scale=b_r1)
        # p_decay_r = sp.stats.gamma.pdf(r_d, a=a_r2, loc=0, scale=b_r2)
        p_r2l_r = sp.stats.gamma.pdf(r_rl, a=a_r3, loc=0, scale=b_r3)
        p_sigma2 = sp.stats.gamma.pdf(sigma2, a=a_sig, loc=0, scale=b_sig)
        p_init = sp.stats.multivariate_normal.pdf(iron_0,mean=iron_0_true, cov = [[0.2, 0],[0, 0.1]])
        lprior = np.log(p_uptake_r*p_r2l_r*p_sigma2*p_init) # *p_decay_r
        return lprior
    # returns a scalart
    return -np.inf

def log_likelihood(theta, times):
    # it is recommended by the EMCEE documentation to not pass data as a local variable. This speeds up parallelisation process
    sigma2, *model_params = theta
    Sigma = sigma2 * np.identity(2)
    *ode_params, iron_r, iron_l = model_params
    model_output = odeint(ode_iron_leaf_root, [iron_r, iron_l], times, args=(ode_params,), rtol=0.000001,atol=0.000001)
    # compute likelihood for all replicates
    logl_all = 0
    for _, Rep in enumerate(Replicates):
        y = y_fe[Rep].copy()
        d_y = np.transpose(model_output)[:,SamplingIndeces] - y # make sure dimensions match
        # logl_all -= 0.5 * (np.trace(np.transpose(d_y) @ np.linalg.inv(Sigma) @ d_y) + len(times) * np.linalg.det(Sigma))
        for iTime in range(len(SamplingIndeces)):
            logl_all += sp.stats.multivariate_normal.logpdf(d_y[:,iTime], mean=None, cov=Sigma, allow_singular=False)
    if math.isnan(logl_all):
        logl_all = -np.inf
    return  logl_all

def ode_iron_leaf_root(x, t, params):
    # the models takes as an input:
    # rate of uptake from matrix to roots, r_mr
    # rate of uptake from roots to leaves, r_rl
    # rate of decay, r_d
    # carrying capacity of the root matrix scaled by the matrix weight, c_max
    r_mr, r_rl = params
    # the updated states are passed as x
    fe_root, fe_leaf = x
    # get the values of dry weight and matrix concentration from the approximating functions
    dr_w = dry_weight(t)/1000 # dry weight should be in kg too!
    fe_in_m = matrix_content(t)
    r_dec_r = rate_decay_r(t)/1000 # increase in weight of roots in kg
    r_dec_l = rate_decay_l(t)/1000 # increase in weight of leafs in kg
    # matrix_weight = 10 #matrix weight in gramms to bring everything to the same measurement units
    dxdt = [(r_mr*dr_w*fe_in_m) - r_rl * fe_root - r_dec_r * fe_root, \
            r_rl * fe_root - r_dec_l * fe_leaf]
    return dxdt

# def ode_iron_leaf_root(x, t, params):
#     # the models takes as an input:
#     # rate of uptake from matrix to roots, r_mr
#     # rate of uptake from roots to leaves, r_rl
#     # rate of decay, r_d
#     # carrying capacity of the root matrix, c_max
#     r_mr, r_rl, r_d, c_max = params
#     # the updated states are passed as x
#     fe_root, fe_leaf = x
#     # get the values of dry weight and matrix concentration from the approximating functions
#     dr_w = dry_weight(t)
#     fe_in_m = matrix_content(t)
#     # matrix_weight = 10 #matrix weight in gramms to bring everything to the same measurement units
#     dxdt = [(r_mr*dr_w*fe_in_m)/(c_max - fe_in_m) - r_rl * fe_root - r_d * fe_root, \
#             r_rl * fe_root - r_d * fe_leaf]
#     return dxdt

# Main
if __name__ == '__main__':
    # enter for how many experiments to run EMCMC
    nExperiments = 1
    # read the iron data here
    # extract the data
    # extract sample ids and locations
    xl_all = pd.ExcelFile('iron_data/all_iron_data_corrected.xlsx')
    df_plants = xl_all.parse('Iron_in_plants')
    df_matrix = xl_all.parse('Iron_in_matrix')
    df_plants[["Day", "Part", "BioRep"]] = df_plants['Samples'].str.split('-', expand=True)
    df_matrix[["Day", "BioRep"]] = df_matrix['Samples'].str.split('-M-', expand=True)
    df_plants.drop(['Samples'], axis=1, inplace=True)
    df_matrix.drop(['Samples'], axis=1, inplace=True)
    # create measurement vectors for iron content
    Iron = [col for col in df_plants.columns if 'iron' in col] + ["Day", "Part", "BioRep"]
    Dry_weight = [col for col in df_plants.columns if 'dw' in col] + ["Day", "BioRep"]
    df_dw = df_plants[Dry_weight].copy()
    df_fe = df_plants[Iron].copy()
    del df_plants
    # average the iron in matrix across replicates -- we need it as a single input
    times = np.linspace(0, 21, 211)
    SamplingDays = [0, 2, 4, 7, 9, 11, 14, 16, 18, 21]
    # we need indeces that correspond to sampling days to select only those points for likelihood
    SamplingIndeces = [i for i, e in enumerate(times) if e in SamplingDays]
    # dry weight and cobtent in perlite are averaged across replicates
    # can also try running the model for each dw series as an alternative
    df_matrix = df_matrix.groupby(['Day']).mean()
    df_dw_r = df_dw.loc[(df_fe['Part'] == 'R'),:].groupby(['Day']).mean()
    df_dw_r_diff = df_dw_r.diff()/2  # beacuse the increase happens over two day time increment
    df_dw_l = df_dw.loc[(df_fe['Part'] == 'L'), :].groupby(['Day']).mean()
    df_dw_l_diff = df_dw_l.diff()/2  # beacuse the increase happens over two day time increment
    ###################################################################################################################
    # Analysis 1: fit the model to each experiment individually, compare posteriors among the models
    # run EMCMC for sets of walkers sampled around these values
    # create measurement vectors and input vectors to be used in the model
    Replicates = df_fe.BioRep.unique()
    Parts = np.flip(df_fe.Part.unique())
    # Extract measurements for one experiment and fit a model to it
    for iExperiment in range(nExperiments):
        Experiment = df_fe.columns[iExperiment]
        y_fe = dict.fromkeys(Replicates)
        iron_0_true = []
        for _, Pt in enumerate(Parts):
            init_val_kg = df_fe.loc[((df_fe['Part'] == Pt) & (df_fe['Day'] == 'D0')), Experiment].mean()
            iron_0_true.append(init_val_kg / 1000)
        # create measurments of fe in plant parts for all replicates
        for _, Rep in enumerate(Replicates):
            # create a measurement matrix
            list_y = []
            for _, Pt in enumerate(Parts):
                # since the dry weight is in grams divied all iron content by 100 to bring to unit: mg/g
                y = df_fe.loc[((df_fe['Part'] == Pt) & (df_fe['BioRep'] == Rep)),Experiment].values/1000
                list_y.append(y)
            y_fe[Rep] = np.stack(tuple(list_y))
            print(y_fe[Rep])
        # create an interpolator for the dry weight and iron content in the matrix
        dry_weight = sp.interpolate.interp1d(SamplingIndeces, df_dw_r.iloc[:,iExperiment].values, fill_value='extrapolate')
        matrix_content = sp.interpolate.interp1d(SamplingIndeces, df_matrix.iloc[:,iExperiment].values,  fill_value='extrapolate')
        rate_decay_r = sp.interpolate.interp1d(SamplingIndeces[1:], df_dw_r_diff.iloc[1:, iExperiment].values,fill_value='extrapolate')
        rate_decay_l = sp.interpolate.interp1d(SamplingIndeces[1:], df_dw_l_diff.iloc[1:, iExperiment].values,fill_value='extrapolate')
    ###################################################################################################################
        ###################################################################################################################
        # run EMCMC for sets of walkers sampled around these values
        # test the model: sigma2 + 3 rates + carrying capacity + initial conditions
        sig2 = .1
        params_0 = [sig2] + [.001, .001] + iron_0_true
        sigma_vector = np.array([0.0001, 0.00001, 0.00001, .01, 0.001])
        covar = np.identity(len(params_0)) * sigma_vector  # create the matrix with diagonal elements defined by sigma_vector
        ndim = len(params_0)
        nwalkers = ndim * 2
        # initialise walkers
        initial_walker_position = np.random.multivariate_normal(mean=params_0, cov=covar, size=nwalkers)
        fileName = "../IronMCMCwalkers/MCMC_iron_experiment"+str(iExperiment)+".h5"
        backend = emcee.backends.HDFBackend(fileName)
        backend.reset(nwalkers, ndim)
        # Prepare placeholders for checking autocorrelation time
        max_iter = 100000
        index_tau = 0
        autocorr = np.empty(max_iter)
        old_tau = np.inf
        # set up a parallel pool to run MCMC
        ncpu = cpu_count()
        ncores = 8
        # print("{0} CPUs".format(ncpu))
        with mp.get_context('fork').Pool(processes=min(ncores, ncpu)) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(times,), pool=pool, backend=backend)
            for sample in sampler.sample(initial_walker_position, iterations=max_iter, progress=True, store=True):
                # test autocorrelation time every 20th iteration
                if sampler.iteration % 20:
                    continue
                # Compute the autocorrelation time so far
                tau = sampler.get_autocorr_time(tol=0)  # tol=0 to get untrustworthy estimate
                autocorr[index_tau] = np.mean(tau)
                index_tau += 1
                # Check convergence
                converged = np.all(tau * 20 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.0001)
                if converged:
                    break
                old_tau = tau
        # samples = sampler.get_chain()
        # tau = sampler.get_autocorr_time(tol=0)
        # print("tau: {0}".format(tau))
        # burnin = int(2 * np.max(tau))
        # thin = int(0.5 * np.min(tau))
        # flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        # log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
        # log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)
        # print("burn-in: {0}".format(burnin))
        # print("thin: {0}".format(thin))
        # print("flat chain shape: {0}".format(flat_samples.shape))
        # print("flat log prob shape: {0}".format(log_prob_samples.shape))

