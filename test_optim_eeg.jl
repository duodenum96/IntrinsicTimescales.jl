# Debugging scratchpad
using Infiltrator
using SciMLSensitivity
using OptimizationOptimJL
import DifferentialEquations as deq
using Statistics
using Zygote
using BayesianINT
using StaticArrays
using Retry
using NaNStatistics

f = (du, u, p, t) -> du .= -u ./ p[1]
g = (du, u, p, t) -> du .= 1.0

function generative_model(theta, times, prob, data_mean, data_sd)
    tau = theta[1]
    freq = theta[2]
    coeff = theta[3]

    prob2 = deq.remake(prob, p=SVector(tau))

    sol = deq.solve(prob2, deq.SOSRI(); verbose=false, saveat=times)
    sol_array = Array(sol)
    num_trials = size(sol_array, 1)

    time_mat = repeat(times, 1, num_trials)'
    phases = rand(num_trials, 1) * 2π
    oscil = sqrt(2.0) * sin.(phases .+ 2π * freq * time_mat)
    data = sqrt(coeff) * oscil .+ sqrt(1.0 - coeff) * sol_array

    data = (data .- mean(data, dims=2)) ./ std(data, dims=2)
    data_scaled = data_sd * data .+ data_mean

    return data_scaled
end

function summary_acf(data, nlags)
    ac = [comp_ac_fft(data[i, :]) for i in axes(data, 1)]
    mean_ac = mean(ac)
    return mean_ac[1:nlags]
end

function summary_psd(data, fs, freqlims)
    psd, freq = comp_psd(data, fs)
    freq_idx = findall(freqlims[1] .<= freq .<= freqlims[2])
    mean_psd = mean(psd[:, freq_idx], dims=1)
    return mean_psd
end

function loss_acf(theta, optparameters)
    mean_ac_target = optparameters[1]
    nlags = optparameters[2]
    times = optparameters[3]
    prob = optparameters[4]
    data_mean = optparameters[5]
    data_sd = optparameters[6]
    data = generative_model(theta, times, prob, data_mean, data_sd)
    mean_ac = summary_acf(data, nlags)
    return sum(abs2, mean_ac .- mean_ac_target)
end

function loss_psd(theta, optparameters)
    mean_psd_target = optparameters[1]
    fs = optparameters[2]
    prob = optparameters[3]
    data_mean = optparameters[4]
    data_sd = optparameters[5]
    freqlims = optparameters[6]
    data = generative_model(theta, times, prob, data_mean, data_sd)
    mean_psd = summary_psd(data, fs, freqlims)
    return sum(abs2, log10.(mean_psd) .- log10.(mean_psd_target)) # logarithmic distance function
end

function optimize(true_tau, coeff)
    dt = 1.0
    T = 10.0 * 1000.0
    numTrials = 30 # 30 x 10 second trials = 5 minutes of data
    freq = 10.0 / 1000.0 # 10 Hz alpha oscillation
    fs = 1.0 / dt
    
    times = collect(dt:dt:T)
    u0 = randn(numTrials)
    prob_outer = deq.SDEProblem(f, g, u0, (0.0, T), true_tau)
    data_mean = 0.0
    data_sd = 1.0
    theta = SVector(true_tau, freq, coeff)
    data = generative_model(theta, times, prob_outer, data_mean, data_sd)
    acf = summary_acf(data, size(data, 2))
    _, freqs = comp_psd(data, fs)
    freqlims = [1.0, 100.0] ./ 1000.0
    freq_idx = findall(freqlims[1] .<= freqs .<= freqlims[2])
    freqs = freqs[freq_idx]
    psd = summary_psd(data, fs, freqlims)
    nlags = floor(Int, 1.1 * acw0(0.0:length(acf), acf))
    lags = collect(0.0:dt:T)[1:nlags]
    acf = acf[1:nlags]
    expdecay_estimate = fit_expdecay(lags, acf)
    knee_freq, osc_peak = fooof_fit(psd[:], freqs)
    knee_estimate = tau_from_knee(knee_freq)

    optparameters_acf = [acf, nlags, times, prob_outer, data_mean, data_sd]
    optparameters_psd = [psd, fs, prob_outer, data_mean, data_sd, freqlims]

    lb = [0.0, 0.0, 0.0]
    ub = [Inf, Inf, 1.0]


    acw50_estimate = tau_from_acw50(acw50(lags, acf))

    u0_opt_acf = MVector(expdecay_estimate, osc_peak, 0.5)
    u0_opt_psd = MVector(knee_estimate, osc_peak, 0.5)
    
    optf_acf = OptimizationFunction(loss_acf, Optimization.AutoForwardDiff())
    optf_psd = OptimizationFunction(loss_psd, Optimization.AutoForwardDiff())
    optprob_acf = OptimizationProblem(optf_acf, u0_opt_acf, optparameters_acf, lb=lb, ub=ub)
    optprob_psd = OptimizationProblem(optf_psd, u0_opt_psd, optparameters_psd, lb=lb, ub=ub)
    @repeat 20 try # Avoid A > B errors of Optim
        global time_elapsed = @elapsed global sol = Optimization.solve(optprob_acf, Optim.LBFGS(), allow_f_increases=true)
        println("Optimization Solution: $(sol.u)")
        acw50_estimate = tau_from_acw50(acw50(lags, acf))
        println("ACW50 estimate: $(acw50_estimate)")
        println("Exp decay estimate: $(u0_opt_acf[1])")
        acfresult = [sol, time_elapsed]
    catch e
        acfresult = [NaN, NaN]
    end

    @repeat 20 try # Avoid A > B errors of Optim
        global time_elapsed = @elapsed global sol = Optimization.solve(optprob_psd, Optim.LBFGS(), allow_f_increases=true)
        println("Optimization Solution: $(sol.u)")
        println("ACW50 estimate: $(acw50_estimate)")
        println("Exp decay estimate: $(u0_opt_acf[1])")
        psdresult = [sol, time_elapsed]
    catch e
        psdresult = [NaN, NaN]
    end

    return acfresult, psdresult, acw50_estimate, expdecay_estimate, knee_estimate
end

# Define parameter ranges
eeg_acw50 = [25.0, 50.0, 75.0, 100.0] # milliseconds
coeffs = [0.1, 0.3, 0.5, 0.7, 0.9] # Coefficient of oscillation
true_tau_values = [-i / log(0.5) for i in eeg_acw50]

# Initialize results arrays
n_taus = length(true_tau_values)
n_coeffs = length(coeffs) 
n_sims = 200

# Create 3D arrays to store results [tau, coeff, sim]
results = (
    acf_solutions = Array{Any}(undef, n_taus, n_coeffs, n_sims),
    psd_solutions = Array{Any}(undef, n_taus, n_coeffs, n_sims),
    acf_times = Array{Float64}(undef, n_taus, n_coeffs, n_sims),
    psd_times = Array{Float64}(undef, n_taus, n_coeffs, n_sims),
    acw50_estimates = Array{Float64}(undef, n_taus, n_coeffs, n_sims),
    true_expdecays = Array{Float64}(undef, n_taus, n_coeffs, n_sims),
    knee_estimates = Array{Float64}(undef, n_taus, n_coeffs, n_sims),
    optim_acf_taus = Array{Float64}(undef, n_taus, n_coeffs, n_sims),
    optim_psd_taus = Array{Float64}(undef, n_taus, n_coeffs, n_sims)
)

# Run simulations
for sim in 1:n_sims
    for (i, true_tau) in enumerate(true_tau_values)
        for (j, coeff) in enumerate(coeffs)
            # Run optimization
            acf_sol, psd_sol, acw50_est, expdecay_est, knee_est = optimize(true_tau, coeff)
            
            # Store results
            results.acf_solutions[i,j,sim] = acf_sol[1]
            results.psd_solutions[i,j,sim] = psd_sol[1]
            results.acf_times[i,j,sim] = acf_sol[2]
            results.psd_times[i,j,sim] = psd_sol[2]
            results.acw50_estimates[i,j,sim] = acw50_est
            results.true_expdecays[i,j,sim] = expdecay_est
            results.knee_estimates[i,j,sim] = knee_est
            results.optim_acf_taus[i,j,sim] = acf_sol[1].u[1]
            results.optim_psd_taus[i,j,sim] = psd_sol[1].u[1]
        end
    end
end

# using Plots

# # Create subplots for each true tau value
# p1 = plot(layout=(2,2), size=(800,600))

# for (i, true_tau) in enumerate(true_tau_vals)
#     histogram!(p1[i], optim_solutions[i], alpha=0.5, label="Optimization", 
#                title="True τ = $(true_tau/1000.0)s")
#     histogram!(p1[i], true_expdecays[i], alpha=0.5, label="Exponential Decay Fitting")
#     histogram!(p1[i], acw50_estimates[i], alpha=0.5, label="ACW50")
#     vline!(p1[i], [true_tau], label="True Tau", color=:black, linewidth=2)
# end

# savefig(p1, "fMRI_optimization.png")

# # Time elapsed plot
# p2 = plot(layout=(2,2), size=(800,600))
# for (i, true_tau) in enumerate(true_tau_vals)
#     histogram!(p2[i], time_elapseds[i], alpha=0.5, label="Elapsed Time (s)",
#                title="True τ = $(true_tau/1000.0)s")
# end

# savefig(p2, "fMRI_optimization_time.png")
# # Calculate RMSE for each simulation
# rmse_optim_sims = [zeros(length(true_tau_vals)) for _ in 1:200]
# rmse_expdecay_sims = [zeros(length(true_tau_vals)) for _ in 1:200]
# rmse_acw50_sims = [zeros(length(true_tau_vals)) for _ in 1:200]

# for sim in 1:200
#     for (i, true_tau) in enumerate(true_tau_vals)
#         rmse_optim_sims[sim][i] = abs(optim_solutions[i][sim] - true_tau) / 1000.0
#         rmse_expdecay_sims[sim][i] = abs(true_expdecays[i][sim] - true_tau) / 1000.0
#         rmse_acw50_sims[sim][i] = abs(acw50_estimates[i][sim] - true_tau) / 1000.0
#     end
# end

# # Create RMSE comparison plots for each tau value
# p3 = plot(layout=(2,2), size=(800,600))

# for (i, true_tau) in enumerate(true_tau_vals)
#     rmse_optim_i = [rmse_optim_sims[sim][i] for sim in 1:200]
#     rmse_expdecay_i = [rmse_expdecay_sims[sim][i] for sim in 1:200]
#     rmse_acw50_i = [rmse_acw50_sims[sim][i] for sim in 1:200]
    
#     histogram!(p3[i], rmse_optim_i, alpha=0.5, label="Optimization",
#                title="RMSE Distribution: τ = $(true_tau/1000.0)s")
#     histogram!(p3[i], rmse_expdecay_i, alpha=0.5, label="Exponential Decay")
#     histogram!(p3[i], rmse_acw50_i, alpha=0.5, label="ACW50")
#     xlabel!("RMSE")
#     ylabel!("Count")
# end

# savefig(p3, "fMRI_optimization_mse.png")
# ###############################################
# # Create scatter plots comparing true vs estimated tau values for each method
# p4 = plot(layout=(1,3), size=(1200,400))

# # Get overall y limits
# all_estimates = vcat(
#     [optim_solutions[i]/1000.0 for i in eachindex(true_tau_vals)]...,
#     [true_expdecays[i]/1000.0 for i in eachindex(true_tau_vals)]...,
#     [acw50_estimates[i]/1000.0 for i in eachindex(true_tau_vals)]...
# )
# ylims_max = (0.0, nanmaximum(all_estimates))

# # Plot optimization results
# for (i, true_tau) in enumerate(true_tau_vals)
#     estimates = optim_solutions[i]/1000.0
#     μ, σ = nanmean(estimates), nanstd(estimates)
#     scatter!(p4[1], fill(true_tau/1000.0, length(estimates)), estimates,
#              alpha=0.3, label=nothing, color=1)
#     scatter!([true_tau/1000.0], [μ], yerror=σ, color=:black, 
#              markersize=10, label=nothing)
# end
# plot!(p4[1], identity, 0:0.5:10, color=:black, linestyle=:dash, label="y=x")
# xlabel!(p4[1], "True τ (s)")
# ylabel!(p4[1], "Estimated τ (s)")
# title!(p4[1], "Optimization")
# ylims!(p4[1], ylims_max)

# # Plot exponential decay results
# for (i, true_tau) in enumerate(true_tau_vals)
#     estimates = true_expdecays[i]/1000.0
#     μ, σ = nanmean(estimates), nanstd(estimates)
#     scatter!(p4[2], fill(true_tau/1000.0, length(estimates)), estimates,
#              alpha=0.3, label=nothing, color=2)
#     scatter!(p4[2], [true_tau/1000.0], [μ], yerror=σ, color=:black,
#              markersize=10, label=nothing, markerfacecolor=:black)
# end
# plot!(p4[2], identity, 0:0.5:10, color=:black, linestyle=:dash, label="y=x")
# xlabel!(p4[2], "True τ (s)")
# ylabel!(p4[2], "Estimated τ (s)")
# title!(p4[2], "Exponential Decay")
# ylims!(p4[2], ylims_max)

# # Plot ACW50 results
# for (i, true_tau) in enumerate(true_tau_vals)
#     estimates = acw50_estimates[i]/1000.0
#     μ, σ = nanmean(estimates), nanstd(estimates)
#     scatter!(p4[3], fill(true_tau/1000.0, length(estimates)), estimates,
#              alpha=0.3, label=nothing, color=3)
#     scatter!(p4[3], [true_tau/1000.0], [μ], yerror=σ, color=:black,
#              markersize=10, label=nothing, markerfacecolor=:black)
# end
# plot!(p4[3], identity, 0:0.5:10, color=:black, linestyle=:dash, label="y=x")
# xlabel!(p4[3], "True τ (s)")
# ylabel!(p4[3], "Estimated τ (s)")
# title!(p4[3], "ACW50")
# ylims!(p4[3], ylims_max)

# savefig(p4, "fMRI_optimization_scatter.png")
