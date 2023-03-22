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
    # carrying capacity of the root matrix, c_max
    # the scaling parameter characterising fe content in the root matrix, fe_matrix_scale
    r_mr, r_rl, r_d, c_max = params
    # the updated states are passed as x
    fe_root, fe_leaf = x
    # get the values of dry weight and matrix concentration from the approximating functions
    dr_w = dry_weight(t)
    fe_in_m = matrix_content(t)
    matrix_weight = 30 #matrix weight in grmms to bring everything to the same measurement units
    dxdt = [(r_mr*dr_w*(fe_in_m/matrix_weight))/(c_max - (fe_in_m/matrix_weight)) - r_rl * fe_root - r_d * fe_root, \
            r_rl * fe_root - r_d * fe_leaf]
    return dxdt

# Main
if __name__ == '__main__':
    # enter for how many experiments to run EMCMC
    nExperiments = 7
    # read the iron data here
    # extract the data
    # extract sample ids and locations
    xl_all = pd.ExcelFile('iron_data/all_iron_data.xlsx')
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
    times = np.linspace(0, 25, 251)
    SamplingDays = [0, 2, 4, 7, 9, 11, 14, 16, 18, 21]
    # we need indeces that correspond to sampling days to select only those points for likelihood
    SamplingIndeces = [i for i, e in enumerate(times) if e in SamplingDays]
    # dry weight and cobtent in perlite are averaged across replicates
    # can also try running the model for each dw series as an alternative
    df_matrix = df_matrix.groupby(['Day']).mean()
    df_dw = df_dw.groupby(['Day']).mean()
    ###################################################################################################################
    # Analysis 1: extract all MCMC outputs for individual experiments and compare posteriors
    # create measurement vectors and input vectors to be used in the model
    Replicates = df_fe.BioRep.unique()
    Parts = df_fe.Part.unique()
    # Extract measurements for one experiment and fit a model to it
    labels = ["$\sigma^2$", "$rate_{m2r}$", "$rate_{r2l}$", "$rate_{decay}$", "$c_{max}$"]
    labels_simple = ['Sigma','r_m2r','r_r2l','r_d','c_max']
    posteriors = dict.fromkeys(labels_simple)
    for iExperiment in range(nExperiments):
        Experiment = df_fe.columns[iExperiment]
        y_fe = dict.fromkeys(Replicates)
        iron_0 = []
        for _, Pt in enumerate(Parts):
            init_val_kg = df_fe.loc[((df_fe['Part'] == Pt) & (df_fe['Day'] == 'D0')), Experiment].mean()
            iron_0.append(init_val_kg/1000)
        # create measurments of fe in plant parts for all replicates
        for _, Rep in enumerate(Replicates):
            # create a measurement matrix
            list_y = []
            for _, Pt in enumerate(Parts):
                y = df_fe.loc[((df_fe['Part'] == Pt) & (df_fe['BioRep'] == Rep)), Experiment].values/1000
                list_y.append(y)
            y_fe[Rep] = np.stack(tuple(list_y))
        # create an interpolator for the dry weight and iron content in the matrix
        dry_weight = sp.interpolate.interp1d(SamplingIndeces, df_dw.iloc[:, iExperiment].values)
        matrix_content = sp.interpolate.interp1d(SamplingIndeces, df_matrix.iloc[:, iExperiment].values)
        ###################################################################################################################
        fileName = "MCMC_iron_experiment"+str(iExperiment)+".h5"
        sampler = emcee.backends.HDFBackend(fileName)
        samples = sampler.get_chain()
        # plot the walkers paths
        fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
        for iParam in range(len(labels)):
            ax = axes.flatten()[iParam]
            ax.plot(samples[:, :, iParam], alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[iParam])
        axes[-1].set_xlabel("step number")
        plt.tight_layout()
        figName = 'Figures/shed_mcmc_walkers_exp_'+str(iExperiment)+'.png'
        plt.savefig(figName)
        # flatten the sample
        tau = sampler.get_autocorr_time(tol=0)
        print("tau: {0}".format(tau))
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
        flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
        log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)
        print("burn-in: {0}".format(burnin))
        print("thin: {0}".format(thin))
        print("flat chain shape: {0}".format(flat_samples.shape))
        print("flat log prob shape: {0}".format(log_prob_samples.shape))
        # build corner plot for each experiment
        import matplotlib.lines as mlines
        mean_line = mlines.Line2D([], [], color='#2ca02c', label='Mean')
        mean_empirical = np.mean(flat_samples, axis=0)
        fig = corner.corner(
            flat_samples, labels=labels,
            quantiles=[0.16, 0.84], show_titles=True, title_kwargs={"fontsize": 10}, title_fmt='.4f')
        corner.overplot_lines(fig, mean_empirical, color='#2ca02c')
        corner.overplot_points(fig, mean_empirical[None], marker="s", color="#2ca02c")
        plt.legend(handles=[mean_line], bbox_to_anchor=(0., 1.15, 1., .0), loc=8)
        figName = 'Figures/mcmc_corner_exp_'+str(iExperiment)+'.png'
        plt.savefig(figName)
        # plot model output and compare to experimental values:
        fig, axes = plt.subplots(len(Parts), figsize=(10, 7), sharex=True)
        inds = np.random.randint(len(flat_samples), size=100)
        y_mean = odeint(ode_iron_leaf_root, iron_0, times, args=(mean_empirical[1:],))
        for ind in inds:
            sample = flat_samples[ind]
            y_mcmc = odeint(ode_iron_leaf_root, iron_0, times, args=(sample[1:],))
            for iPart in range(len(Parts)):
                axes.flatten()[iPart].plot(times, y_mcmc[:, iPart],color='#1f77b4', alpha=0.1)
        for iPart in range(len(Parts)):
            axes.flatten()[iPart].plot(times, y_mean[:, iPart], color='#ff7f0e', label="MCMC mean")
            for _, Rep in enumerate(Replicates):
                axes.flatten()[iPart].plot(SamplingDays, y_fe[Rep][iPart, :], marker='o', markersize=2, label='Measured')
            axes.flatten()[iPart].legend(fontsize=10)
            axes.flatten()[iPart].set_xlabel("time, days")
            axes.flatten()[iPart].set_ylabel("Fe, mg/kg")
        plt.tight_layout()
        figName = 'Figures/Model_output_exp_'+str(iExperiment)+'.png'
        plt.savefig(figName)
        ################################################################################################################

