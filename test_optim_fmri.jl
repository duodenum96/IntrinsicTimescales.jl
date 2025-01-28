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

function optimize(true_tau)
    dt = 1.0 * 1000.0
    T = 150.0 * 1000.0
    numTrials = 1

    f = (du, u, p, t) -> du .= -u ./ p[1]
    g = (du, u, p, t) -> du .= 1.0
    times_const = collect(dt:dt:T)
    u0 = randn(numTrials)
    prob_outer = deq.SDEProblem(f, g, u0, (0.0, T), true_tau)
    sol_outer = deq.solve(prob_outer, deq.SOSRI(); verbose=false, saveat=times_const)
    sol_array = Array(sol_outer)
    # mean_data = mean(sol_array)
    ac = [comp_ac_fft(sol_array[i, :]) for i in 1:size(sol_array, 1)]
    mean_ac_tmp = mean(ac)
    acw0_estimate = acw0(0.0:length(mean_ac_tmp), mean_ac_tmp)
    nlags = floor(Int, 1.1 * acw0_estimate)
    mean_ac = mean_ac_tmp[1:nlags]
    lags = collect(0.0:dt:T)[1:nlags]
    true_expdecay = fit_expdecay(lags, mean_ac)

    # function grad_loss(G, u, p)
    #     # u: tau
    #     G .= sum( ( (expdecay(2u[1], lags) .* lags) .- (expdecay(u[1], lags) .* lags .* mean_ac) ) ./ u[1]^2 )
    #     return nothing
    # end

    function loss_function(u, optparameters)
        prob = deq.remake(prob_outer, p=u)
        sol = deq.solve(prob, deq.SOSRI(); saveat=times_const)
        sol_array = Array(sol)
        ac = [comp_ac_fft(view(sol_array, i, :), n_lags=nlags) for i in 1:size(sol_array, 1)]
        mean_ac_sim = mean(ac)

        d = sum(abs2, mean_ac_sim .- mean_ac)
        # if !(d isa Float64)
        #     println(d.value)
        # end
        return d
    end

    p = nothing
    u0_opt = MVector(true_expdecay)
    optf = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(optf, u0_opt)
    @repeat 20 try # Avoid A > B errors of Optim
        global time_elapsed = @elapsed global sol = Optimization.solve(optprob, Optim.LBFGS(), allow_f_increases=true)
        println("Optimization Solution: $(sol.u)")
        acw50_estimate = tau_from_acw50(acw50(lags, mean_ac))
        println("ACW50 estimate: $(acw50_estimate)")
        println("Exp decay estimate: $(u0_opt[1])")
        return sol, time_elapsed, acw50_estimate, true_expdecay
    catch e
        return NaN, NaN, NaN, NaN
    end

end

true_tau_vals = [3.0, 5.0, 7.0, 9.0] * 1000.0 # milliseconds

zaas = [[] for _ in true_tau_vals]
time_elapseds = [[] for _ in true_tau_vals]
acw50_estimates = [[] for _ in true_tau_vals]
true_expdecays = [[] for _ in true_tau_vals]
optim_solutions = [[] for _ in true_tau_vals]

for i in 1:200
    for (i, true_tau) in enumerate(true_tau_vals)
        zaa, time_elapsed, acw50_estimate, true_expdecay = optimize(SVector(true_tau))
        push!(zaas[i], zaa)
        push!(time_elapseds[i], time_elapsed)
        push!(acw50_estimates[i], acw50_estimate)
        push!(true_expdecays[i], true_expdecay)
        if !(zaa isa Float64)
            push!(optim_solutions[i], zaa.u[1])
        else
            push!(optim_solutions[i], zaa)
        end
    end
end

using Plots

# Create subplots for each true tau value
p1 = plot(layout=(2,2), size=(800,600))

for (i, true_tau) in enumerate(true_tau_vals)
    histogram!(p1[i], optim_solutions[i], alpha=0.5, label="Optimization", 
               title="True τ = $(true_tau/1000.0)s")
    histogram!(p1[i], true_expdecays[i], alpha=0.5, label="Exponential Decay Fitting")
    histogram!(p1[i], acw50_estimates[i], alpha=0.5, label="ACW50")
    vline!(p1[i], [true_tau], label="True Tau", color=:black, linewidth=2)
end

savefig(p1, "fMRI_optimization.png")

# Time elapsed plot
p2 = plot(layout=(2,2), size=(800,600))
for (i, true_tau) in enumerate(true_tau_vals)
    histogram!(p2[i], time_elapseds[i], alpha=0.5, label="Elapsed Time (s)",
               title="True τ = $(true_tau/1000.0)s")
end

savefig(p2, "fMRI_optimization_time.png")
# Calculate RMSE for each simulation
rmse_optim_sims = [zeros(length(true_tau_vals)) for _ in 1:200]
rmse_expdecay_sims = [zeros(length(true_tau_vals)) for _ in 1:200]
rmse_acw50_sims = [zeros(length(true_tau_vals)) for _ in 1:200]

for sim in 1:200
    for (i, true_tau) in enumerate(true_tau_vals)
        rmse_optim_sims[sim][i] = abs(optim_solutions[i][sim] - true_tau) / 1000.0
        rmse_expdecay_sims[sim][i] = abs(true_expdecays[i][sim] - true_tau) / 1000.0
        rmse_acw50_sims[sim][i] = abs(acw50_estimates[i][sim] - true_tau) / 1000.0
    end
end

# Create RMSE comparison plots for each tau value
p3 = plot(layout=(2,2), size=(800,600))

for (i, true_tau) in enumerate(true_tau_vals)
    rmse_optim_i = [rmse_optim_sims[sim][i] for sim in 1:200]
    rmse_expdecay_i = [rmse_expdecay_sims[sim][i] for sim in 1:200]
    rmse_acw50_i = [rmse_acw50_sims[sim][i] for sim in 1:200]
    
    histogram!(p3[i], rmse_optim_i, alpha=0.5, label="Optimization",
               title="RMSE Distribution: τ = $(true_tau/1000.0)s")
    histogram!(p3[i], rmse_expdecay_i, alpha=0.5, label="Exponential Decay")
    histogram!(p3[i], rmse_acw50_i, alpha=0.5, label="ACW50")
    xlabel!("RMSE")
    ylabel!("Count")
end

savefig(p3, "fMRI_optimization_mse.png")
###############################################
# Create scatter plots comparing true vs estimated tau values for each method
p4 = plot(layout=(1,3), size=(1200,400))

# Get overall y limits
all_estimates = vcat(
    [optim_solutions[i]/1000.0 for i in eachindex(true_tau_vals)]...,
    [true_expdecays[i]/1000.0 for i in eachindex(true_tau_vals)]...,
    [acw50_estimates[i]/1000.0 for i in eachindex(true_tau_vals)]...
)
ylims_max = (0.0, nanmaximum(all_estimates))

# Plot optimization results
for (i, true_tau) in enumerate(true_tau_vals)
    estimates = optim_solutions[i]/1000.0
    μ, σ = nanmean(estimates), nanstd(estimates)
    scatter!(p4[1], fill(true_tau/1000.0, length(estimates)), estimates,
             alpha=0.3, label=nothing, color=1)
    scatter!([true_tau/1000.0], [μ], yerror=σ, color=:black, 
             markersize=10, label=nothing)
end
plot!(p4[1], identity, 0:0.5:10, color=:black, linestyle=:dash, label="y=x")
xlabel!(p4[1], "True τ (s)")
ylabel!(p4[1], "Estimated τ (s)")
title!(p4[1], "Optimization")
ylims!(p4[1], ylims_max)

# Plot exponential decay results
for (i, true_tau) in enumerate(true_tau_vals)
    estimates = true_expdecays[i]/1000.0
    μ, σ = nanmean(estimates), nanstd(estimates)
    scatter!(p4[2], fill(true_tau/1000.0, length(estimates)), estimates,
             alpha=0.3, label=nothing, color=2)
    scatter!(p4[2], [true_tau/1000.0], [μ], yerror=σ, color=:black,
             markersize=10, label=nothing, markerfacecolor=:black)
end
plot!(p4[2], identity, 0:0.5:10, color=:black, linestyle=:dash, label="y=x")
xlabel!(p4[2], "True τ (s)")
ylabel!(p4[2], "Estimated τ (s)")
title!(p4[2], "Exponential Decay")
ylims!(p4[2], ylims_max)

# Plot ACW50 results
for (i, true_tau) in enumerate(true_tau_vals)
    estimates = acw50_estimates[i]/1000.0
    μ, σ = nanmean(estimates), nanstd(estimates)
    scatter!(p4[3], fill(true_tau/1000.0, length(estimates)), estimates,
             alpha=0.3, label=nothing, color=3)
    scatter!(p4[3], [true_tau/1000.0], [μ], yerror=σ, color=:black,
             markersize=10, label=nothing, markerfacecolor=:black)
end
plot!(p4[3], identity, 0:0.5:10, color=:black, linestyle=:dash, label="y=x")
xlabel!(p4[3], "True τ (s)")
ylabel!(p4[3], "Estimated τ (s)")
title!(p4[3], "ACW50")
ylims!(p4[3], ylims_max)

savefig(p4, "fMRI_optimization_scatter.png")
