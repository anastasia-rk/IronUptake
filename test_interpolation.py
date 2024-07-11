import numpy as np
from setup import *
from MCMC_iron import *
import corner
plt.rcParams['figure.figsize'] = (20,10)
plt.rcParams['figure.dpi'] = 200
plt.style.use("ggplot")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# Definitions
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
    fe_in_m = matrix_content(t)/0.02 # has to be per kg to align the measurement units.
    r_dec_r = rate_decay_r(t)/1000 # increase in weight of roots in kg
    r_dec_l = rate_decay_l(t)/1000 # increase in weight of leafs in kg
    # matrix_weight = 10 #matrix weight in gramms to bring everything to the same measurement units
    dxdt = [(r_mr*fe_in_m) - r_rl * fe_root - r_dec_r * fe_root, \
            r_rl * fe_root - r_dec_l * fe_leaf]
    return dxdt
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
    nExperiments = 7
    # read the iron data here
    # extract the data
    # extract sample ids and locations
    xl_all = pd.ExcelFile('iron_data/all_iron_data.xlsx')
    df_plants = xl_all.parse('Iron_in_plants')
    df_matrix = xl_all.parse('Iron_dose')
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
    fig2,axes2 = plt.subplots(nExperiments,2,sharex=True)
    fig1,axes1 = plt.subplots(nExperiments,2,sharex=True)
    ###################################################################################################################
    # Analysis 1: extract all MCMC outputs for individual experiments and compare posteriors
    # create measurement vectors and input vectors to be used in the model
    Replicates = df_fe.BioRep.unique()
    Parts = np.flip(df_fe.Part.unique())
    # Extract measurements for one experiment and fit a model to it
    # labels = ["$\sigma^2$", "$rate_{m2r}$", "$rate_{r2l}$", "$rate_{decay}$", "$c_{max}$", '$w_{matrix}$']
    labels = ["$\sigma^2$", "$rate_{m2r}$", "$rate_{r2l}$", "$Fe_{0}(root)$", "$Fe_{0}(leaf)$"]
    labels_simple = ['$\sigma^2$','$r_m2r$','$r_r2l$', 'Fe_{0}(R)','Fe_{0}(L)']
    # create dataframes to store posterior properties
    dfs = []
    for iParam in range(len(labels)):
        dfs.append(pd.DataFrame(columns=['Experiment', 'Mean', 'Variance', 'Skewness', 'Kurtosis']))
    posteriors = dict.fromkeys(labels_simple)
    for iExperiment in range(nExperiments):
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
        dry_weight = sp.interpolate.interp1d(SamplingDays, df_dw_r.iloc[:, iExperiment].values, kind='cubic',
                                             fill_value='extrapolate')
        matrix_content = sp.interpolate.interp1d(SamplingDays, df_matrix.iloc[:, iExperiment].values, kind='cubic',
                                                 fill_value='extrapolate')
        rate_decay_r = sp.interpolate.interp1d(SamplingDays[1:], df_dw_r_diff.iloc[1:, iExperiment].values, kind='cubic',
                                               fill_value=(df_dw_r_diff.iloc[1, iExperiment], df_dw_r_diff.iloc[-1, iExperiment]),bounds_error=False)
        rate_decay_l = sp.interpolate.interp1d(SamplingDays[1:], df_dw_l_diff.iloc[1:, iExperiment].values,
                                               fill_value=(df_dw_l_diff.iloc[1, iExperiment], df_dw_l_diff.iloc[-1, iExperiment]),\
                                               kind='cubic',bounds_error=False)
        # plot average rate of Fe decay due to plant growth
        axes2[iExperiment,0].plot(SamplingDays[1:], df_dw_r_diff.iloc[1:, iExperiment].values, 'o-', color='#d95f02',
                         label='Exp.' + str(iExperiment + 1),alpha=0.8)
        axes2[iExperiment,0].plot(times, rate_decay_r(times), 'k--',alpha=0.5,label='B-spline fit')
        axes2[iExperiment,0].set_ylabel('$r_{decay}$')
        axes2[iExperiment,1].plot(SamplingDays[1:], df_dw_l_diff.iloc[1:, iExperiment].values, 'o-', color='#d95f02',
                         label='Exp.' + str(iExperiment + 1),alpha=0.8)
        axes2[iExperiment,1].plot(times, rate_decay_l(times), 'k--',alpha=0.5,label='B-spline fit')
        axes2[iExperiment,1].set_ylabel('$r_{decay}$')
        axes2[iExperiment,0].legend(loc="lower left", ncol=2)
        # plot dry weight of roots and iron concentration in the matrix
        axes1[iExperiment,0].plot(SamplingDays, df_dw_r.iloc[:, iExperiment].values, 'o-', color='#d95f02',
                         label='Exp.' + str(iExperiment + 1),alpha=0.8)
        axes1[iExperiment,0].plot(times, dry_weight(times), 'k--',alpha=0.5,label='B-spline fit')
        axes1[iExperiment,0].set_ylabel('$dw_{roots}$, g')
        axes1[iExperiment,1].plot(SamplingDays, df_matrix.iloc[:, iExperiment].values, 'o-', color='#d95f02',
                         label='Exp.' + str(iExperiment + 1),alpha=0.8)
        axes1[iExperiment,1].plot(times, matrix_content(times), 'k--',alpha=0.5,label='B-spline fit')
        axes1[iExperiment,1].set_ylabel('$Fe_{m}$, mg')
        axes1[iExperiment,1].legend(loc="lower right", ncol=2)
    # save dry weight and matrix concentration figure
    plt.figure(fig1.number)
    axes1[0, 0].set_title('Dry weight of roots')
    axes1[0, 1].set_title('Iron in the matrix')
    axes1[-1, 0].set_xlabel('Time, days')
    axes1[-1, 1].set_xlabel('Time, days')
    plt.tight_layout(pad=0.5)
    figName = 'Figures/dry_weight_interp.png'
    plt.savefig(figName, dpi=600)
    # save the rates of decay figure
    plt.figure(fig2.number)
    axes2[0, 0].set_title('In roots, g/day')
    axes2[0, 1].set_title('In leaves, g/day')
    axes2[-1, 0].set_xlabel('Time,days')
    axes2[-1, 1].set_xlabel('Time,days')
    plt.tight_layout(pad=0.5)
    figName = 'Figures/rates_of_decay.png'
    plt.savefig(figName, dpi=600)