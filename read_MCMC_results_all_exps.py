import numpy as np
from setup import *
# from MCMC_iron import *
import corner
plt.rcParams['figure.figsize'] = (20,10)
plt.rcParams['figure.dpi'] = 200
plt.style.use("ggplot")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

# Definitions:
########################################################################################################################
# models:

# no inhibition:
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
    dr_w = dry_weight(t) # dry weight should be in g too!
    fe_in_m = matrix_content(t) # has to be per g to align the measurement units.
    r_dec_r = rate_decay_r(t) # increase in weight of roots in g
    r_dec_l = rate_decay_l(t) # increase in weight of leafs in g
    # matrix_weight = 10 #matrix weight in gramms to bring everything to the same measurement units
    dxdt = [(r_mr*fe_in_m) - r_rl * fe_root - r_dec_r * fe_root, \
            r_rl * fe_root - r_dec_l * fe_leaf]
    return dxdt

# # max capacity:
# def ode_iron_leaf_root(x, t, params):
#     # the models takes as an input:
#     # rate of uptake from matrix to roots, r_mr
#     # rate of uptake from roots to leaves, r_rl
#     # rate of decay, r_d
#     # carrying capacity of the root matrix scaled by the matrix weight, c_max
#     r_mr, r_rl, c_max = params
#     # the updated states are passed as x
#     fe_root, fe_leaf = x
#     # get the values of dry weight and matrix concentration from the approximating functions
#     dr_w = dry_weight(t) # dry weight should be in g too!
#     fe_in_m = matrix_content(t) # has to be per g to align the measurement units.
#     r_dec_r = rate_decay_r(t) # increase in weight of roots in g
#     r_dec_l = rate_decay_l(t) # increase in weight of leafs in g
#     # matrix_weight = 10 #matrix weight in gramms to bring everything to the same measurement units
#     dxdt = [(r_mr*dr_w*fe_in_m)*(c_max - fe_root)/c_max - r_rl * fe_root - r_dec_r * fe_root, \
#             r_rl * fe_root - r_dec_l * fe_leaf]
#     return dxdt

# # michaelis constant:
# def ode_iron_leaf_root(x, t, params):
#     # the models takes as an input:
#     # rate of uptake from matrix to roots, r_mr
#     # rate of uptake from roots to leaves, r_rl
#     # rate of decay, r_d
#     # carrying capacity of the root matrix scaled by the matrix weight, c_max
#     r_mr, r_rl, c_min, MMc = params
#     # the updated states are passed as x
#     fe_root, fe_leaf = x
#     # get the values of dry weight and matrix concentration from the approximating functions
#     dr_w = dry_weight(t)  # dry weight should be in g too!
#     fe_in_m = matrix_content(t)  # has to be per kg to align the measurement units.
#     r_dec_r = rate_decay_r(t)  # increase in weight of roots in g
#     r_dec_l = rate_decay_l(t)  # increase in weight of leafs in g
#     # matrix_weight = 10 #matrix weight in gramms to bring everything to the same measurement units
#     dxdt = [(r_mr*dr_w*(fe_in_m - c_min))/(fe_in_m + MMc - c_min) - r_rl * fe_root - r_dec_r * fe_root, \
#             r_rl * fe_root - r_dec_l * fe_leaf]
#     return dxdt

# # time-invariant decay rate:
# def ode_iron_leaf_root(x, t, params):
#     # the models takes as an input:
#     # rate of uptake from matrix to roots, r_mr
#     # rate of uptake from roots to leaves, r_rl
#     # rate of decay, r_d
#     # carrying capacity of the root matrix scaled by the matrix weight, c_max
#     r_mr, r_rl, r_d, MMc = params
#     # the updated states are passed as x
#     fe_root, fe_leaf = x
#     # get the values of dry weight and matrix concentration from the approximating functions
#     dr_w = dry_weight(t)   # dry weight should be in g too!
#     fe_in_m = matrix_content(t)  # has to be per g to align the measurement units.
#     # matrix_weight = 10 #matrix weight in gramms to bring everything to the same measurement units
#     dxdt = [(r_mr*dr_w*fe_in_m)/(MMc + fe_in_m) - r_rl * fe_root - r_d * fe_root, \
#             r_rl * fe_root - r_d * fe_leaf]
#     return dxdt
########################################################################################################################
# autocorr time computation:
def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))
    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n
    # Optionally normalize
    if norm:
        acf /= acf[0]
    return acf

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

# Main
if __name__ == '__main__':
    # enter for how many experiments to run EMCMC
    Experiments = range(4,7)
    nExperiments = len(Experiments)
    # read the iron data here
    # extract the data
    # extract sample ids and locations
    xl_all = pd.ExcelFile('iron_data/all_iron_data_corrected.xlsx')
    df_plants = xl_all.parse('Iron_in_plants')
    # df_matrix = xl_all.parse('Iron_in_matrix')
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
    df_dw_r = df_dw.loc[(df_fe['Part'] == 'R'), :].groupby(['Day']).mean()
    df_dw_r_diff = df_dw_r.diff() / 2  # beacuse the increase happens over two day time increment
    df_dw_l = df_dw.loc[(df_fe['Part'] == 'L'), :].groupby(['Day']).mean()
    df_dw_l_diff = df_dw_l.diff() / 2  # beacuse the increase happens over two day time increment
    # create a figure for rates of decay in roots and leaves
    fig2,axes2 = plt.subplots(2,1,sharex=True)
    ###################################################################################################################
    # Analysis 1: extract all MCMC outputs for individual experiments and compare posteriors
    # create measurement vectors and input vectors to be used in the model
    Replicates = df_fe.BioRep.unique()
    Parts = np.flip(df_fe.Part.unique())
    # Extract measurements for one experiment and fit a model to it
    # Model 4 - no inhibition
    labels = ["$\sigma^2$", "$rate_{m2r}$", "$rate_{r2l}$","$Fe_{0}(root)$", "$Fe_{0}(leaf)$"]
    labels_simple = ['$\sigma^2$','$r_m2r$','$r_r2l$', 'Fe_{0}(R)','Fe_{0}(L)']
    # # Model 3 - max capacity of roots
    # labels = ["$\sigma^2$", "$rate_{m2r}$", "$rate_{r2l}$", '$c_{max}$',"$Fe_{0}(root)$", "$Fe_{0}(leaf)$"]
    # labels_simple = ['$\sigma^2$','$r_m2r$','$r_r2l$', '$c_{max}$', 'Fe_{0}(R)','Fe_{0}(L)']
    # # Model 2 - min cocentration in matrix & MM constant
    # labels = ["$\sigma^2$", "$rate_{m2r}$", "$rate_{r2l}$",'$c_{max}$', '$MM const$',"$Fe_{0}(root)$", "$Fe_{0}(leaf)$"]
    # labels_simple = ['$\sigma^2$','$r_m2r$','$r_r2l$','$c_{max}$', 'MM const', 'Fe_{0}(R)','Fe_{0}(L)']
    # # Model 1 -  const decay rate
    # labels = ["$\sigma^2$", "$rate_{m2r}$", "$rate_{r2l}$", "$rate_{decay}$", 'MM const',"$Fe_{0}(root)$", "$Fe_{0}(leaf)$"]
    # labels_simple = ['$\sigma^2$','$r_m2r$','$r_r2l$', "$r_{decay}$", 'MM const', 'Fe_{0}(R)','Fe_{0}(L)']
    # # create dataframes to store posterior properties
    ###################################################################################################################
    fileName = "../IronMCMCwalkers/MCMC_iron_all_experiments.h5"
    sampler = emcee.backends.HDFBackend(fileName)
    samples = sampler.get_chain()
    # flatten the chain
    tau = sampler.get_autocorr_time(tol=0)
    print("tau: {0}".format(tau))
    burnin = int(2 * np.max(tau))
    thin = max(1, int(0.5 * np.min(tau)))
    # plot the walkers paths
    fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
    for iParam in range(len(labels)):
        ax = axes.flatten()[iParam]
        ax.plot(samples[:, :, iParam], alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[iParam])
        ax.text(0.99, 0.85, r'$\tau=$' + "{:.2f}".format(tau[iParam]), horizontalalignment='right',
                transform=ax.transAxes, fontsize=10)
    axes[-1].set_xlabel("step number")
    plt.tight_layout()
    figName = 'Figures/mcmc_walkers_all_experiments.png'
    plt.savefig(figName)
    flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
    log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)
    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))
    print("flat chain shape: {0}".format(flat_samples.shape))
    print("flat log prob shape: {0}".format(log_prob_samples.shape))
    # build corner plot for each experiment
    import matplotlib.lines as mlines
    mean_line = mlines.Line2D([], [], color='#2ca02c', label='Empirical mean')
    dummyline = mlines.Line2D([], [], color='k', linestyle='--', label='16 and 84 percentiles')
    mean_empirical = np.mean(flat_samples, axis=0)
    fig = corner.corner(
        flat_samples, labels=labels,
        quantiles=[0.16, 0.84], show_titles=True, title_quantiles=None, title_kwargs={"fontsize": 10}, title_fmt='.4f')
    corner.overplot_lines(fig, mean_empirical, color='#2ca02c')
    corner.overplot_points(fig, mean_empirical[None], marker="s", color="#2ca02c")
    plt.legend(handles=[mean_line, dummyline], bbox_to_anchor=(0., 1.15, 1., .0), loc=8)
    figName = 'Figures/mcmc_corner_exp_all_experiments.png'
    plt.savefig(figName)
    ###################################################################################################################
    dfs = []
    for iParam in range(len(labels)):
        dfs.append(pd.DataFrame(columns=['Experiment', 'Mean', 'Variance', 'Skewness', 'Kurtosis']))
    posteriors = dict.fromkeys(labels_simple)
    for iExperiment in Experiments:
        Experiment = df_fe.columns[iExperiment]
        y_fe = dict.fromkeys(Replicates)
        # create measurments of fe in plant parts for all replicates
        for _, Rep in enumerate(Replicates):
            # create a measurement matrix
            list_y = []
            for _, Pt in enumerate(Parts):
                y = df_fe.loc[((df_fe['Part'] == Pt) & (df_fe['BioRep'] == Rep)), Experiment].values/1000
                list_y.append(y)
            y_fe[Rep] = np.stack(tuple(list_y))
        # create an interpolator for the dry weight and iron content in the matrix
        dry_weight = sp.interpolate.interp1d(SamplingDays, df_dw_r.iloc[:, iExperiment].values,  fill_value='extrapolate')
        matrix_content = sp.interpolate.interp1d(SamplingDays, df_matrix.iloc[:, iExperiment].values,  fill_value='extrapolate')
        rate_decay_r = sp.interpolate.interp1d(SamplingDays[1:], df_dw_r_diff.iloc[1:, iExperiment].values,
                                               kind='cubic',
                                               fill_value=(
                                               df_dw_r_diff.iloc[1, iExperiment], df_dw_r_diff.iloc[-1, iExperiment]),
                                               bounds_error=False)
        rate_decay_l = sp.interpolate.interp1d(SamplingDays[1:], df_dw_l_diff.iloc[1:, iExperiment].values,
                                               fill_value=(
                                               df_dw_l_diff.iloc[1, iExperiment], df_dw_l_diff.iloc[-1, iExperiment]), \
                                               kind='cubic', bounds_error=False)
        # plot model output and compare to experimental values:
        fig, axes = plt.subplots(len(Parts), figsize=(10, 7), sharex=True)
        inds = np.random.randint(len(flat_samples), size=100)
        y_mean = odeint(ode_iron_leaf_root,  mean_empirical[-2:], times, args=(mean_empirical[1:-2],))
        for ind in inds:
            sample = flat_samples[ind]
            y_mcmc = odeint(ode_iron_leaf_root, sample[-2:], times, args=(sample[1:-2],))
            for iPart in range(len(Parts)):
                axes.flatten()[iPart].plot(times, y_mcmc[:, iPart],color='#1f77b4', alpha=0.1)
        for iPart in range(len(Parts)):
            axes.flatten()[iPart].plot(times, y_mean[:, iPart], color='#ff7f0e', label="MCMC mean")
            for _, Rep in enumerate(Replicates):
                labelRep = 'Replicate ' + Rep
                axes.flatten()[iPart].plot(SamplingDays, y_fe[Rep][iPart, :], marker='o', markersize=2, label=labelRep)
            axes.flatten()[iPart].legend( ncol=2, fontsize=10)
            axes.flatten()[iPart].set_ylabel("Fe in "+ Parts[iPart] +", mg/g")
        axes.flatten()[-1].set_xlabel("Time, days")
        plt.tight_layout()
        figName = 'Figures/Model_output_exp_'+str(iExperiment+1)+'.png'
        plt.savefig(figName)
        ################################################################################################################
#         compute summary statistics of each posterior distribution
        variance = np.var(flat_samples, axis=0)
        skewness = sp.stats.skew(flat_samples, axis=0, bias=True)
        kurtosis = sp.stats.kurtosis(flat_samples, axis=0, bias=True)
        for iParam in range(len(labels)):
            df_row = pd.DataFrame({"Experiment": [iExperiment+1],"Mean": [mean_empirical[iParam]],"Variance":[variance[iParam]],\
                                         "Skewness":[skewness[iParam]],"Kurtosis":[kurtosis[iParam]]})
            dfs[iParam] = pd.concat([dfs[iParam], df_row], ignore_index=True)
    ####################################################################################################################
    # save tables into a single excel file
    sheet_labels = ['Sigma squared','Rate of uptake','Rate of transfer','Fe_0 in roots','Fe_0 in leaves']
    with pd.ExcelWriter('Posteriors_all_exp.xlsx') as writer:
        for iParam in range(len(sheet_labels)):
            dfs[iParam].to_excel(writer, sheet_name=sheet_labels[iParam])


