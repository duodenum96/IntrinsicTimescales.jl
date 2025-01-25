using BayesianINT
using BenchmarkTools
using Plots
using LinearAlgebra
BLAS.set_num_threads(1)

"""
fMRI timescales are on the order of 3-10 seconds (ACW-0) (https://www.sciencedirect.com/science/article/pii/S1053811920306273?via%3Dihub#fig1)
EEG/MEG timescales are on the order of 25-100 milliseconds (ACW-50) (https://www.nature.com/articles/s42003-021-01785-z)

Going from ACW-0 to τ is tricky. However, we can use the ACW-50 to calculate τ.
acw50(tau) = -tau * log(0.5)
tau(acw50) = -acw50 / log(0.5)

We'll do two sets of benchmarks:
- Comparison of methods (ACW-50, ACW-0, tau, FOOOF) with BayesianINT method
- Comparison between software implementations (statsmodels acf vs FOOOF vs abcTau vs BayesianINT)
"""

fmri_taus = [3.0, 5.0, 7.0, 9.0] * 1000.0 # milliseconds
fmri_acw50 = acw50_analytical.(fmri_taus)
eeg_acw50 = [25.0, 50.0, 75.0, 100.0] # milliseconds
eeg_taus = [-i / log(0.5) for i in eeg_acw50]

fmri_dt = 1000.0 # TR=1s
eeg_dt = 1 / 500.0 # 500 Hz sampling rate

fmri_ntrials = 1 # Just one run
eeg_ntrials = 30 # 30 trials of 10 seconds ==> 300 seconds

fmri_T = 300.0 * 1000 # 300 seconds
eeg_T = 10.0 * 1000 # 10 seconds (per trial)

# Compare ACW-50, ACW-0, tau, FOOOF methods with BayesianINT method
# Start with fMRI
nsims = 30
acw50_naive = zeros(length(fmri_taus), nsims)
acw0_naive = zeros(length(fmri_taus), nsims)
tau_naive = zeros(length(fmri_taus), nsims)
fooof_naive = zeros(length(fmri_taus), nsims)

n_accepted = 100
all_thetas = zeros(length(fmri_taus), nsims, n_accepted)
tau_bINT = zeros(length(fmri_taus), nsims)
N_MAP = 10000

for i in eachindex(fmri_taus)
    i_tau = fmri_taus[i]
    Threads.@threads for j in 1:nsims
        ou = generate_ou_process(i_tau, 1.0, fmri_dt, fmri_T, fmri_ntrials)
        acf = comp_ac_time(ou, size(ou, 2))
        lags = collect((0:length(acf)-1) * fmri_dt)
        global acw50_naive[i, j] = acw50(lags, acf)
        global acw0_naive[i, j] = acw0(lags, acf)
        global tau_naive[i, j] = fit_expdecay(lags, acf)
        power, freq = comp_psd(ou, 1/fmri_dt)
        global fooof_naive[i, j] = 1 / fooof_fit(power, freq; oscillation_peak=false) # tau = 1 / knee

        model = OneTimescaleModel(ou, "informed", acf, 1.0, fmri_dt, fmri_T, fmri_ntrials, 1.0, floor(Int, 2*acw0_naive[i, j] / 1000))
        results = pmc_abc(model, epsilon_0=model.epsilon, max_iter=10000, min_accepted=n_accepted, steps=40, target_epsilon=0.001)
        theta_accepted = results[end].theta_accepted
        global tau_bINT[i, j] = find_MAP(theta_accepted, N_MAP)[1]
        global all_thetas[i, j, :] = theta_accepted
        println("i = $i, j = $j")
        flush(stdout)
    end
    # println("i = $i, j = $j")
    # flush(stdout)
end

tau_bINT_mean = mean(tau_bINT ./ 1000.0, dims=2)
tau_bINT_std = std(tau_bINT ./ 1000.0, dims=2)
println(tau_bINT_mean)
println(tau_bINT_std)

acw50_naive_mean = mean(acw50_naive ./ 1000.0, dims=2)
acw50_naive_std = std(acw50_naive ./ 1000.0, dims=2)
println(acw50_naive_mean)
println(acw50_naive_std)

acw0_naive_mean = mean(acw0_naive ./ 1000.0, dims=2)
acw0_naive_std = std(acw0_naive ./ 1000.0, dims=2)
println(acw0_naive_mean)
println(acw0_naive_std)

fooof_naive_mean = mean(fooof_naive ./ 1000.0, dims=2)
fooof_naive_std = std(fooof_naive ./ 1000.0, dims=2)
println(fooof_naive_mean)
println(fooof_naive_std)

tau_naive_mean = mean(tau_naive ./ 1000.0, dims=2)
tau_naive_std = std(tau_naive ./ 1000.0, dims=2)
println(tau_naive_mean)
println(tau_naive_std)

scatter(fmri_taus ./ 1000.0, tau_bINT_mean[:], yerr=tau_bINT_std, label="BayesianINT", legend=:bottomright)
scatter!(fmri_taus ./ 1000.0, acw50_naive_mean, yerr=acw50_naive_std, label="ACW-50")
scatter!(fmri_taus ./ 1000.0, acw0_naive_mean, yerr=acw0_naive_std, label="ACW-0")
scatter!(fmri_taus ./ 1000.0, fooof_naive_mean, yerr=fooof_naive_std, label="FOOOF")
scatter!(fmri_taus ./ 1000.0, tau_naive_mean, yerr=tau_naive_std, label="statsmodels")

