
import numpy as np
import matplotlib.pyplot as plt

def get_master_params(file_paths):
	main_means = []
	main_sds = []
	for path in file_paths:
		mean_sd = np.load(path)
		main_means.append(mean_sd[0])
		main_sds.append(mean_sd[1])
	return [np.array(main_means), np.array(main_sds)]

def rel_error(truth, approx):
	return np.sum((truth - approx) ** 2) / np.sum(truth ** 2) 

N = 1000
p = 500
snr = 1

induce_paths50 = []
for i in range(0, 1990, 10):
	induce_paths50.append('../summary_stats/dump/fitc_master_params_N_{0}_p_{1}_scale_{2}_induce_{3}_iter_{4}.npy'.format(N, p, snr, 50, i))

induce_paths100 = []
for i in range(0, 1990, 10):
	induce_paths100.append('../summary_stats/dump/fitc_master_params_N_{0}_p_{1}_scale_{2}_induce_{3}_iter_{4}.npy'.format(N, p, snr, 100, i))

induce_paths200 = []
for i in range(0, 1990, 10):
	induce_paths200.append('../summary_stats/dump/fitc_master_params_N_{0}_p_{1}_scale_{2}_induce_{3}_iter_{4}.npy'.format(N, p, snr, 200, i))

induce_paths500 = []
for i in range(0, 1990, 10):
	induce_paths500.append('../summary_stats/dump/fitc_master_params_N_{0}_p_{1}_scale_{2}_induce_{3}_iter_{4}.npy'.format(N, p, snr, 500, i))

master_params_induce50 = get_master_params(induce_paths50) 
master_params_induce100 = get_master_params(induce_paths100) 
master_params_induce200 = get_master_params(induce_paths200) 
master_params_induce500 = get_master_params(induce_paths500) 
all_master_params_induce = [master_params_induce50, master_params_induce100, master_params_induce200, master_params_induce500]
master_params_exact = np.load('../summary_stats/exact_master_params_N_1000_p_500_scale_1.npy')

master_params_subsamp50 = np.load('../summary_stats/subsamp_master_params_N_1000_p_500_scale_1_induce_50.npy')
master_params_subsamp100 = np.load('../summary_stats/subsamp_master_params_N_1000_p_500_scale_1_induce_100.npy')
master_params_subsamp200 = np.load('../summary_stats/subsamp_master_params_N_1000_p_500_scale_1_induce_200.npy')
master_params_subsamp500 = np.load('../summary_stats/subsamp_master_params_N_1000_p_500_scale_1_induce_500.npy')
all_master_params_sub = [master_params_subsamp50, master_params_subsamp100, master_params_subsamp200, master_params_subsamp500]

mean_exact = master_params_exact[0].mean(axis=0)
sd_exact = master_params_exact[1].mean(axis=0)

all_mean_sub = []
all_sd_sub = []
for i in range(4):
	all_mean_sub.append(all_master_params_sub[i][0].mean(axis=0))
	all_sd_sub.append(all_master_params_sub[i][1].mean(axis=0))

all_mean_induce = []
all_sd_induce = []
for i in range(4):
	all_mean_induce.append(all_master_params_induce[i][0].mean(axis=0))
	all_sd_induce.append(np.sqrt(all_master_params_induce[i][1]).mean(axis=0))

mean_errors_sub = []
sd_errors_sub = []
for i in range(4):
	mean_errors_sub.append(rel_error(mean_exact, all_mean_sub[i]))
	sd_errors_sub.append(rel_error(sd_exact, all_sd_sub[i]))

mean_errors_induce = []
sd_errors_induce = []

for i in range(4):
	mean_errors_induce.append(rel_error(mean_exact, all_mean_induce[i]))
	sd_errors_induce.append(rel_error(sd_exact, all_sd_induce[i]))

induce_times = [31973.824817419052, 36679.067974090576, 40766.06842136383, 50755.75591301918]
subsamp_times = [3862.021940946579, 7606.321288347244, 10100.687921762466, 18693.579425573349]
induce_arr = [50, 100, 200, 500]
time_exact = 51784.64281320572

# Mean error 
plt.figure(num=None, figsize=(3.5, 3), dpi=150)
plt.plot(subsamp_times, mean_errors_sub, marker='o', label='Subsample', color='red')
plt.plot(induce_times, mean_errors_induce, marker='o', label='FITC', color='green')
plt.axvline(time_exact, linestyle='--', color='black', label='Exact')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Rel. Error', size=18)
plt.xlabel('Time (s)', size=18)
plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig('../icml_2019_paper/bayes_interactions/images/mean_error.png')
plt.close()

# SD error 
plt.figure(num=None, figsize=(3.5, 3), dpi=150)
plt.plot(subsamp_times, sd_errors_sub, marker='o', label='Subsample', color='red')
plt.plot(induce_times, sd_errors_induce, marker='o', label='FITC', color='green')
plt.axvline(time_exact, linestyle='--', color='black', label='Exact')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Rel. Error', size=18)
plt.xlabel('Time (s)', size=18)
plt.legend(fontsize=11)
plt.tight_layout()
plt.tight_layout()
plt.savefig('../icml_2019_paper/bayes_interactions/images/sd_error.png')
plt.close()

################# Kernel Hyperparams #################
params_exact = load_pymc3_run('../model/exact_N_{0}_p_{1}_scale_{2}.pkl'.format(1000, 500, 1))[0]
params_induce = [load_pymc3_run('../model/fitc_N_{0}_p_{1}_scale_{2}_induce_{3}.pkl'.format(1000, 500, 1, n_induce))[0] for n_induce in induce_arr]
params_sub = [load_pymc3_run('../model/subsamp_N_{0}_p_{1}_scale_{2}_induce_{3}.pkl'.format(1000, 500, 1, n_induce))[0] for n_induce in induce_arr]

exact_kappa = params_exact['kappa'].mean(axis=0)
induce_kappa = [params['kappa'].mean(axis=0) for params in params_induce]
sub_kappa = [params['kappa'].mean(axis=0) for params in params_sub]

induce_kappa_errors = [rel_error(exact_kappa, induce_kappa[i]) for i in range(len(induce_arr))]
sub_kappa_errors = [rel_error(exact_kappa, sub_kappa[i]) for i in range(len(induce_arr))]

# Mean error 
plt.figure(num=None, figsize=(3.5, 3), dpi=150)
plt.plot(subsamp_times, sub_kappa_errors, marker='o', label='Subsample', color='red')
plt.plot(induce_times, induce_kappa_errors, marker='o', label='FITC', color='green')
plt.axvline(time_exact, linestyle='--', color='black', label='Exact')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Rel. Error', size=18)
plt.xlabel('Time (s)', size=18)
plt.legend(fontsize=13)
plt.tight_layout()
plt.savefig('../icml_2019_paper/bayes_interactions/images/kernel_hyp_error.png')
plt.close()

plt.figure(num=None, figsize=(3, 3), dpi=150)
plt.scatter(range(500), sub_kappa[0],color='red', label='Subsample50', alpha=.7)
plt.scatter(range(500), exact_kappa, color='blue', label='Exact', alpha=.7)
plt.scatter(range(500), induce_kappa[0], color='green', label='FITC50', alpha=.7)
plt.ylabel('Local Scale Mean', size=18)
plt.xlabel('Covariate Index', size=18)
plt.axvline(5, linestyle='--', color='black')
plt.yscale('log')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('../icml_2019_paper/bayes_interactions/images/manhat50.png')
plt.close()

plt.figure(num=None, figsize=(3, 3), dpi=150)
plt.scatter(range(500), sub_kappa[3],color='red', label='Subsample500', alpha=.7)
plt.scatter(range(500), exact_kappa, color='blue', label='Exact', alpha=.7)
plt.scatter(range(500), induce_kappa[3], color='green', label='FITC500', alpha=.7)
plt.ylabel('Local Scale Mean', size=18)
plt.xlabel('Covariate Index', size=18)
plt.axvline(5, linestyle='--', color='black')
plt.yscale('log')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('../icml_2019_paper/bayes_interactions/images/manhat500.png')
plt.close()

################# ESS per Second #################
exact_samp_stats = load_pymc3_run('../model/exact_N_{0}_p_{1}_scale_{2}.pkl'.format(1000, 500, 1))[1]
exact_samp_stats = exact_samp_stats['n_eff'].values
fitc50_samp_stats = load_pymc3_run('../model/fitc_N_{0}_p_{1}_scale_{2}_induce_{3}.pkl'.format(1000, 500, 1, 50))[1]
fitc50_samp_stats = fitc50_samp_stats['n_eff'].values
fitc500_samp_stats = load_pymc3_run('../model/fitc_N_{0}_p_{1}_scale_{2}_induce_{3}.pkl'.format(1000, 500, 1, 500))[1]
fitc500_samp_stats = fitc500_samp_stats['n_eff'].values

sub50_samp_stats = load_pymc3_run('../model/subsamp_N_{0}_p_{1}_scale_{2}_induce_{3}.pkl'.format(1000, 500, 1, 50))[1]
sub50_samp_stats = sub50_samp_stats['n_eff'].values
sub500_samp_stats = load_pymc3_run('../model/subsamp_N_{0}_p_{1}_scale_{2}_induce_{3}.pkl'.format(1000, 500, 1, 500))[1]
sub500_samp_stats = sub500_samp_stats['n_eff'].values

plt.figure(num=None, figsize=(3, 3), dpi=150)
plt.boxplot([exact_samp_stats, fitc50_samp_stats, fitc500_samp_stats, sub50_samp_stats, sub500_samp_stats], labels=['Exact', 'FITC50', 'FITC500', 'Sub50', 'Sub500'])
plt.yscale('log')
plt.xticks(fontsize=14, rotation=90)
plt.ylabel('Eff. Sample Size', size=14)
plt.savefig('../icml_2019_paper/bayes_interactions/images/ess.png')
plt.close()
