import Turing as tr
using BayesianINT
import DifferentialEquations as deq
using Statistics
using Plots
using DifferentiationInterfaceTest
using LinearAlgebra
using Zygote
using DifferentiationInterface
using SciMLSensitivity

# tr.setadbackend(:zygote)
true_tau = 300.0 / 1000.0
num_trials = 30
T = 10.0
dt = 1 / 500
times = dt:dt:T
nlags = 150

f = (du, u, p, t) -> du .= -u ./ p[1]
g = (du, u, p, t) -> du .= sqrt(2.0 / p[1])
p = true_tau
u0 = randn(num_trials) # Quick hack instead of ensemble problem
prob = deq.SDEProblem(f, g, u0, (0.0, T), p)
sol = deq.solve(prob, deq.SOSRI(); saveat=times)
data_ts = Array(sol)
data_acf = comp_ac_time_adfriendly(data_ts, nlags)

"""
data: ACF
prob: SciML style SDEProblem
"""
tr.@model function fit_acf(data, prob)
    # Prior
    tau ~ tr.truncated(tr.Normal(0.5, 0.5), 0.0, Inf)
    σ ~ tr.Exponential(1)
    p2 = [tau]
    
    prob_re = deq.remake(prob, p=p2)
    predicted = deq.solve(prob_re, deq.SOSRI(); saveat=times)

    predicted_ts = Array(predicted)
    predicted_acf = comp_ac_time_adfriendly(predicted_ts, nlags)
    # predicted_acf = comp_ac_fft(predicted_ts; n_lags=nlags)

    # Likelihood
    for i in eachindex(data)
        data[i] ~ tr.Normal(predicted_acf[i], σ^2)
    end

    return nothing
end

model = fit_acf(data_acf, prob)
advi = tr.ADVI(10, 1000, AutoZygote())
chain = tr.vi(model, advi)
# chain_hmc = tr.sample(model, tr.NUTS(), tr.MCMCSerial(), 1000, 1; progress=true)
inferred_params = (mean(rand(chain, 4000); dims=2)...,)
inferred_params_var = (std(rand(chain, 4000); dims=2)...,)

tau_draws = rand(chain, 4000)[1, :][:]
tau_selected = rand(tau_draws, 20)

plot(1:nlags, data_acf, label="Data", linewidth=2)
predicted_acfs = zeros(length(tau_selected), nlags)
for (i, tau_pred) in enumerate(tau_selected)
    sol = deq.solve(prob, deq.SOSRI(); saveat=times, p=[tau_pred])
    predicted_ts = Array(sol)
    predicted_acfs[i,:] = comp_ac_time_adfriendly(predicted_ts, nlags)
end

lower_bound = [quantile(predicted_acfs[:,i], 0.025) for i in 1:nlags]
upper_bound = [quantile(predicted_acfs[:,i], 0.975) for i in 1:nlags]
plot!(1:nlags, [lower_bound upper_bound], fillrange=upper_bound, fillalpha=0.3, label="95% CI", color=:blue)

histogram(tau_draws)
vline!([true_tau])
