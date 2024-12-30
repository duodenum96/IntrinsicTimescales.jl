# src/models/one_timescale.jl

module OneTimescale

using Distributions
using ..Models
using ..OrnsteinUhlenbeck
using BayesianINT

export OneTimescaleModel

"""
One-timescale OU process model
"""
struct OneTimescaleModel <: AbstractTimescaleModel
    data::Matrix{Float64}
    prior::Vector{Distribution}
    data_sum_stats::Vector{Float64}
    epsilon::Float64
    dt::Float64
    T::Float64
    numTrials::Int
    data_var::Float64
    n_lags::Int
end

# Implementation of required methods
function Models.generate_data(model::OneTimescaleModel, theta)
    tau = theta
    return generate_ou_process(tau, model.data_var, model.dt, model.T, model.numTrials; backend="sciml")
end

function Models.summary_stats(model::OneTimescaleModel, data)
    return comp_ac_fft(data; n_lags=model.n_lags)
end

function Models.distance_function(model::OneTimescaleModel, sum_stats, data_sum_stats)
    return linear_distance(sum_stats, data_sum_stats)
end

end # module OneTimescale 