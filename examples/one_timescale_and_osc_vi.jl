import Turing as tr
using ForwardDiff
using BayesianINT
using Statistics
using Plots
using LinearAlgebra
using DifferentiationInterface
using SciMLSensitivity
import DifferentialEquations as deq

# Set true parameters
true_tau = 300.0 / 1000.0  # timescale in seconds
true_freq = 10.0  # Hz
true_coeff = 0.95  # mixing coefficient
num_trials = 20
T = 10.0
dt = 1 / 125
times = dt:dt:T
nlags = 100
frange = [0, 50]

# Generate synthetic data
true_theta = [true_tau, true_freq, true_coeff]
data_mean = 0.0
data_var = 1.0
data_ts = generate_ou_with_oscillation(true_theta, dt, T, num_trials, data_mean, data_var)
# Replace ACF computation with PSD
fs = 1/dt
data_psd, freq = comp_psd_adfriendly(data_ts, fs)
freq_idx = findall(freq .>= frange[1] .&& freq .<= frange[2])
data_psd = data_psd[freq_idx]
freq = freq[freq_idx]

log_psd = log10.(data_psd)
mean_psd = mean(log_psd)
sd_psd = std(log_psd)
log_psd_scaled = (log_psd .- mean_psd) * sd_psd

"""
Model for fitting PSD of OU process with oscillation
data: PSD values
freq: frequency values
fs: sampling frequency
"""
tr.@model function fit_psd_with_osc(data, freq, fs)
    # Priors
    tau ~ tr.truncated(tr.Normal(0.5, 0.1), 0.0, Inf)
    # freq_osc ~ tr.truncated(tr.Normal(10.0, 5.0), 0.0, Inf)
    # Frequency is on a different scale. We need to transform it.
    freq_osc ~ tr.truncated(tr.Normal(0.5, 0.05), 0.0, Inf)
    coeff ~ tr.truncated(tr.Normal(0.5, 0.1), 0.0, 1.0)
    σ ~ tr.Exponential(1)

    transformed_freq = freq_osc * 20
    
    # Generate predicted data
    theta = [tau, transformed_freq, coeff]
    println("theta: $theta")
    predicted_ts = generate_ou_with_oscillation(theta, dt, T, num_trials, data_mean, data_var)
    predicted_psd, _ = comp_psd_adfriendly(predicted_ts, fs)
    predicted_psd_sel = log10.(predicted_psd[freq_idx])
    # predicted_psd_scaled = (predicted_psd_sel .- mean_psd) * sd_psd

    # Likelihood
    # data ~ tr.MvNormal(predicted_psd_scaled, σ^2)
    data ~ tr.MvNormal(predicted_psd_sel, σ^2)

    return nothing
end

# Fit model using ADVI
model = fit_psd_with_osc(log_psd, freq, fs)
advi = tr.ADVI(10, 300, AutoZygote())
chain = tr.vi(model, advi)

# Get inferred parameters
inferred_params = median(rand(chain, 4000); dims=2)
inferred_params[2] = inferred_params[2] * 20
inferred_params_var = var(rand(chain, 4000); dims=2)

# Generate predictions with inferred parameters
theta_pred = [inferred_params[1], inferred_params[2], inferred_params[3]]
predicted_ts = generate_ou_with_oscillation(theta_pred, dt, T, num_trials, data_mean, data_var)
predicted_psd, _ = comp_psd_adfriendly(predicted_ts, fs)
predicted_psd_sel = predicted_psd[freq_idx]

# Get parameter draws from chain
chain_draws = rand(chain, 4000)
tau_draws = chain_draws[1,:][:]
freq_draws = chain_draws[2,:][:] * 20
coeff_draws = chain_draws[3,:][:]


# Randomly select some parameter combinations for uncertainty visualization
n_samples = 20
sample_indices = rand(1:4000, n_samples)
predicted_psds = zeros(n_samples, length(freq))

for (i, idx) in enumerate(sample_indices)
    theta_sample = [tau_draws[idx], freq_draws[idx], coeff_draws[idx]]
    predicted_ts = generate_ou_with_oscillation(theta_sample, dt, T, num_trials, data_mean, data_var)
    predicted_psd, _ = comp_psd_adfriendly(predicted_ts, fs)
    predicted_psds[i,:] = predicted_psd[freq_idx]
end

# Plot PSD with uncertainty
plot(freq, data_psd, label="Data", linewidth=2, xscale=:log10, yscale=:log10)
lower_bound = [quantile(predicted_psds[:,i], 0.025) for i in 1:length(freq)]
upper_bound = [quantile(predicted_psds[:,i], 0.975) for i in 1:length(freq)]
plot!(freq, [lower_bound upper_bound], fillrange=upper_bound, fillalpha=0.3, label="95% CI", color=:blue)
xlabel!("Frequency (Hz)")
ylabel!("Power")

# Plot parameter histograms
p1 = histogram(tau_draws, title="tau", label="")
vline!(p1, [true_tau], label="True value")

p2 = histogram(freq_draws, title="frequency", label="")
vline!(p2, [true_freq], label="True value")

p3 = histogram(coeff_draws, title="coefficient", label="")
vline!(p3, [true_coeff], label="True value")

plot(p1, p2, p3, layout=(1,3), size=(900,300))
