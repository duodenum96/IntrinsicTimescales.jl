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
fs = 1 / dt
times = dt:dt:T
nlags = 300

f = (du, u, p, t) -> du .= -u ./ p[1]
g = (du, u, p, t) -> du .= sqrt(2.0 / p[1])
p = true_tau
u0 = randn(num_trials) # Quick hack instead of ensemble problem
prob = deq.SDEProblem(f, g, u0, (0.0, T), p)
sol = deq.solve(prob, deq.SOSRI(); saveat=times)
data_ts = Array(sol)
# Make some data missing. Each trial is 10 seconds. Randomly pick 3 second to be missing for each trial.
missing_seconds = 3
# data_ts_missing = convert(Array{Union{Missing, Float64}}, data_ts)
# Create missing mask - each trial has a random 1 second window of missing data
missing_mask = falses(size(data_ts))
samples_per_second = missing_seconds * Int(fs)  # Number of samples in 1 second
for trial in 1:num_trials
    # Randomly select start index for 1 second window
    max_start = length(times) - samples_per_second
    start_idx = rand(1:max_start)
    missing_mask[trial, start_idx:(start_idx + samples_per_second)] .= true
end

data_ts_missing[missing_mask] .= NaN

data_acf = comp_ac_time_missing(data_ts_missing, nlags)
# Compare with the ACF of the non-missing data
data_acf_nonmissing = comp_ac_time_missing(data_ts, nlags)
plot(0:(nlags-1), data_acf, label="Missing data")
plot!(0:(nlags-1), data_acf_nonmissing, label="Non-missing data")

tr.@model function fit_acf(data_acf)
    tau ~ tr.truncated(tr.Normal(0.5, 0.1), 0.0, Inf)
    σ ~ tr.truncated(tr.Exponential(1.0), 0.0, Inf)
    p2 = [tau]
    prob2 = deq.remake(prob, p=p2)
    sol2 = deq.solve(prob2, deq.SOSRI(); saveat=times)
    data_pred = Array(sol2)
    data_pred_missing = convert(Array{Union{Missing, Float64}}, data_pred)
    data_pred_missing[missing_mask] .= missing
    data_acf_pred = comp_ac_time_missing(data_pred_missing, nlags)

    data_acf ~ tr.MvNormal(data_acf_pred, σ^2 * I)
    return nothing
end

model = fit_acf(data_acf)

advi = tr.ADVI(10, 200, AutoZygote())
chain = tr.vi(model, advi)

inferred_params = (mean(rand(chain, 4000); dims=2)...,)
inferred_params_var = (var(rand(chain, 4000); dims=2)...,)

tau_pred = inferred_params[1]
sol = deq.solve(prob, deq.SOSRI(); saveat=times, p=[tau_pred])
predicted_ts = Array(sol)
predicted_acf = comp_ac_time_missing(predicted_ts, nlags)

plot(1:nlags, data_acf, label="Data", linewidth=2, legend=:bottomright)
plot!(1:nlags, predicted_acf, label="Predicted", linewidth=2, legend=:bottomright)
