# Import
from setup import *
from scipy.stats import *

if __name__ == '__main__':
    plt.close('all')
    # plot priors - generate intervals
    uptake_rate_plot = np.arange(0, 1, 0.005)
    root2leaf_rate_plot = np.arange(0, 1, 0.005)
    decay_rate_plot = np.arange(0, 1, 0.005)
    carrying_capacity = np.arange(0, 1000, 10)
    # prior parameters
    a_r1, b_r1 = 5, .01
    a_r2, b_r2 = 5, .001
    a_r3, b_r3 = 5, .01
    p_uptake_r = sp.stats.gamma.pdf(uptake_rate_plot,a=a_r1, loc=0, scale=b_r1)
    p_decay_r = sp.stats.gamma.pdf(decay_rate_plot,a=a_r2, loc=0, scale=b_r2)
    p_r2l_r = sp.stats.gamma.pdf(root2leaf_rate_plot,a=a_r3, loc=0, scale=b_r3)
    p_carryig_capacity = sp.stats.uniform.pdf(carrying_capacity, loc=1, scale=1000)

    # plots
    fig, axes = plt.subplots(2,2, figsize=(10, 7))
    axes.flatten()[0].plot(uptake_rate_plot, p_uptake_r, lw=2, label="$p(r_{matrix2root})$")
    axes.flatten()[0].legend(loc='best', fontsize=14)
    axes.flatten()[0].set_xlabel("$r_{uptake}$")
    axes.flatten()[1].plot(decay_rate_plot, p_decay_r, lw=2, label="$p(r_{decay})$")
    axes.flatten()[1].legend(loc='best', fontsize=14)
    axes.flatten()[1].set_xlabel("$r_{decay}$")
    axes.flatten()[2].plot(root2leaf_rate_plot, p_r2l_r, lw=2, label="$p(r_{root2leaf})$")
    axes.flatten()[2].legend(loc='best', fontsize=14)
    axes.flatten()[2].set_xlabel("$r_{root2leaf}$")
    axes.flatten()[3].plot(carrying_capacity, p_carryig_capacity, lw=2, label="$p(carrying capacity)$")
    axes.flatten()[3].legend(loc='best', fontsize=14)
    axes.flatten()[3].set_xlabel("carrying capacity")
    fig.suptitle('Initial priors for unknown parameters',fontsize=14)
    fig.tight_layout(pad=1.0)
    plt.savefig('Valeria_initial_priors.png')
    plt.show()
    # approximating prior for diffusion rates
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Diffusion rates are analysed and sampled in 1/SECOND!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # compute parameters of the perceived Gamma distribution and check goodness of fit with K-S test
    # d_mean = np.mean(kicheva_diffusions)
    # d_sigma = np.std(kicheva_diffusions)
    # we scale up and down arbitrarily to see how pdf relates to data spread
    # d_alpha = (d_mean)**2/d_sigma/5
    # d_scale = (d_sigma/d_mean) * 4
    # kicheva_range = np.arange(0, 55, .1)
    # gamma_kicheva = sp.stats.gamma.pdf(kicheva_range, a=d_alpha, loc=0, scale=d_scale)
    # test_result_less = sp.stats.kstest(kicheva_diffusions, 'gamma', args=(d_alpha, 0, d_scale), alternative='less')
    # print(test_result_less)
    # if test_result_less.pvalue<0.05:
    #     print('Null hypothesis is rejected. Distribution of data is upper-bounded by chosen Gamma distribution.')
    # else:
    #     print('Null hypothesis is accepted. Distribution of data is not upper-bounded by chosen Gamma distribution.')
    #
    # test_result_greater = sp.stats.kstest(kicheva_diffusions, 'gamma', args=(d_alpha, 0, d_scale), alternative='greater')
    # print(test_result_greater)
    # if test_result_greater.pvalue<0.05:
    #     print('Null hypothesis is rejected. Distribution of data is lower-bounded by chosen Gamma distribution.')
    # else:
    #     print('Null hypothesis is accepted. Distribution of data is not lower-bounded by chosen Gamma distribution.')
    # # plot the histogram of Kicheva data and approximating Gamma distribution
    # fig, axes = plt.subplots(1, 1, figsize=(10,13))
    # factor =np.max(gamma_kicheva)/8
    # axes.hist(kicheva_diffusions,bins=8,histtype='step',weights=factor*np.ones_like(kicheva_diffusions) ,label='Histogram')
    # axes.plot(kicheva_range, gamma_kicheva, label='Approx. Gamma')
    # axes.scatter(kicheva_diffusions,np.zeros_like(kicheva_diffusions)+0.002,marker='o', s=25, color='y', label='Points')
    # axes.legend(loc='best', fontsize=14)
    # fig.tight_layout(pad=1.0)
    # fig.suptitle('Diffusion rate Gamma prior',fontsize=14)
    # plt.savefig('Valeria_diffusion_prior_Kicheva.png')
    # plt.show()
    # # Approximating prior for decay rates
    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # # Decay rates are analysed and sampled in 1/MINUte!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # # compute parameters of the perceived Gamma distribution and check goodness of fit with K-S test
    # nu_mean = np.mean(kicheva_decay_rates)
    # nu_sigma = np.std(kicheva_decay_rates)
    # nu_alpha = (nu_mean) ** 2 / nu_sigma
    # nu_scale = (nu_sigma / nu_mean)
    # nu_range = np.arange(0, 1, .01)
    # gamma_nu = sp.stats.gamma.pdf(nu_range, a=nu_alpha, loc=0, scale=nu_scale)
    # test_result_less = sp.stats.kstest(kicheva_decay_rates, 'gamma', args=(nu_alpha, 0, nu_scale), alternative='less')
    # print(test_result_less)
    # if test_result_less.pvalue < 0.05:
    #     print('Null hypothesis is rejected. Distribution of data is upper-bounded by chosen Gamma distribution.')
    # else:
    #     print('Null hypothesis is accepted. Distribution of data is not upper-bounded by chosen Gamma distribution.')
    #
    # test_result_greater = sp.stats.kstest(kicheva_diffusions, 'gamma', args=(nu_alpha, 0, nu_scale),
    #                                       alternative='greater')
    # print(test_result_greater)
    # if test_result_greater.pvalue < 0.05:
    #     print('Null hypothesis is rejected. Distribution of data is lower-bounded by chosen Gamma distribution.')
    # else:
    #     print('Null hypothesis is accepted. Distribution of data is not lower-bounded by chosen Gamma distribution.')
    #
    # # plot histogram of Kicheva data and perform a
    # fig, axes = plt.subplots(1, 1, figsize=(10, 13))
    # factor = 1
    # axes.hist(kicheva_decay_rates, bins=8,histtype='step', weights=factor * np.ones_like(kicheva_decay_rates), label='Histogram')
    # # axes.plot(nu_range, gamma_nu, label='Approx. Gamma')
    # axes.scatter(kicheva_decay_rates, np.zeros_like(kicheva_decay_rates) + 0.02, marker='o', s=25, color='y',
    #              label='Points')
    # axes.legend(loc='best', fontsize=14)
    # fig.tight_layout(pad=1.0)
    # fig.suptitle('Decay rate Gamma prior',fontsize=14)
    # plt.savefig('Valeria_decay_prior_Kicheva.png')
    # plt.show()
    # Gamma does not approximate the data, we will use uniform prior and use reported values to assesss min and max values: min 0 max 0.002 - that 172.8 1/day!
    nu_min = 0
    nu_range = 0.002
    # # sampling from priors with selected hyper-parameters
    sampleSize = 1000
    sample_ur = sp.stats.gamma.rvs(size=sampleSize,a=a_r1, loc=0, scale=b_r1)
    sample_dr = sp.stats.gamma.rvs(size=sampleSize,a=a_r2, loc=0, scale=b_r2)
    sample_rlr = sp.stats.gamma.rvs(size=sampleSize,a=a_r3, loc=0, scale=b_r3)
    sample_cc = sp.stats.uniform.rvs(size=sampleSize, loc=1, scale=1000)

    # plot sample histograms
    fig, axes = plt.subplots(2,2, figsize=(10, 7))
    axes.flatten()[0].hist(sample_ur,density=True,histtype='step',label='Histogram')
    axes.flatten()[0].scatter(sample_ur, sp.stats.gamma.pdf(sample_ur, a=a_r1, loc=0, scale=b_r1), color='#ff7f0e', lw=0.5, label="$p(r_{matrix2root})$")
    axes.flatten()[0].legend(loc='best', fontsize=14)
    axes.flatten()[0].set_xlabel("uptake rate")
    axes.flatten()[1].hist(sample_dr,density=True,histtype='step',label='Histogram')
    axes.flatten()[1].scatter(sample_dr, sp.stats.gamma.pdf(sample_dr, a=a_r2, loc=0, scale=b_r2), color='#ff7f0e', lw=0.5, label="$p(r_{decay})$")
    axes.flatten()[1].legend(loc='best', fontsize=14)
    axes.flatten()[1].set_xlabel("decay rate")
    axes.flatten()[2].hist(sample_rlr,density=True,histtype='step',label='Histogram')
    axes.flatten()[2].scatter(sample_rlr, sp.stats.gamma.pdf(sample_rlr, a=a_r3, loc=0, scale=b_r3), color='#ff7f0e', lw=0.5, label="$p(r_{root2leaf})$")
    axes.flatten()[2].legend(loc='best', fontsize=14)
    axes.flatten()[2].set_xlabel("rate to leaf rate")
    axes.flatten()[3].hist(sample_cc,density=True,histtype='step',label='Histogram')
    axes.flatten()[3].scatter(sample_cc, sp.stats.uniform.pdf(sample_cc, loc=1, scale=1000), color='#ff7f0e', lw=0.5, label="$p(carrying capacity)$")
    axes.flatten()[3].legend(loc='best', fontsize=14)
    axes.flatten()[3].set_xlabel("carrying capacity")
    fig.suptitle('Samples for unknown parameters',fontsize=14)
    fig.tight_layout(pad=1.0)
    plt.savefig('priors_evaluated_at_sampling_points.png')
    plt.show()

    # # save sample
    header = ['r_0','D','nu_d','lambda']
    # with open("Sampled_params.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(header)
    #     writer.writerows(zip(sample_r0, sample_D, sample_nu, sample_lambda))
