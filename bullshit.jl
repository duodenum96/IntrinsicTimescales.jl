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

function optimize()
    dt = 1.0 * 1000.0
    T = 150.0 * 1000.0
    numTrials = 1
    true_tau = SVector(1.5 * 1000.0)

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
        global time_elapsed = @time global sol = Optimization.solve(optprob, Optim.LBFGS(), allow_f_increases=true)
        println("Optimization Solution: $(sol.u)")
        acw50_estimate = tau_from_acw50(acw50(lags, mean_ac))
        println("ACW50 estimate: $(acw50_estimate)")
        println("Exp decay estimate: $(u0_opt[1])")
        return sol, time_elapsed, acw50_estimate, true_expdecay
    catch e
        return NaN, NaN, NaN, NaN
    end

end

zaas = []
time_elapseds = []
acw50_estimates = []
true_expdecays = []
optim_solutions = []

for i in 1:200
    zaa, time_elapsed, acw50_estimate, true_expdecay = optimize()
    push!(zaas, zaa)
    push!(time_elapseds, time_elapsed)
    push!(acw50_estimates, acw50_estimate)
    push!(true_expdecays, true_expdecay)
    if !(zaa isa Float64)
        push!(optim_solutions, zaa.u[1])
    else
        push!(optim_solutions, zaa)
    end
end

true_tau = 1.5 * 1000.0
using Plots
histogram(optim_solutions, alpha=0.5, label="Optimization")
histogram!(true_expdecays, alpha=0.5, label="Exponential Decay Fitting")
histogram!(acw50_estimates, alpha=0.5, label="ACW50")
vline!([true_tau], label="True Tau", color=:black, linewidth=4)
savefig("fMRI_optimization.png")

histogram(time_elapseds, alpha=0.5, label="Time")
savefig("fMRI_optimization_time.png")

