# src/models/one_timescale.jl

module OneTimescale

using Distributions
using ..Models
using ..OrnsteinUhlenbeck
using BayesianINT

export OneTimescaleModel

function informed_prior(data_sum_stats::Vector{<:Real}, dt::Real, n_lags::Real)
    lags = collect((0.0:(n_lags-1)) * dt)
    tau = fit_expdecay(lags, data_sum_stats)
    return [Normal(tau, 1000)] # Convert to ms
end

"""
One-timescale OU process model
"""
struct OneTimescaleModel <: AbstractTimescaleModel
    data::Matrix{<:Real}
    prior::Vector{Any}
    data_sum_stats::Vector{<:Real}
    epsilon::Real
    dt::Real
    T::Real
    numTrials::Real
    data_var::Real
    n_lags::Real
end

function OneTimescaleModel(data, prior::String, data_sum_stats, epsilon, dt, T, numTrials, data_var, n_lags)
    if prior == "informed"
        if length(data_sum_stats) != n_lags
            data_sum_stats_trunc = data_sum_stats[1:n_lags]
        else
            data_sum_stats_trunc = data_sum_stats
        end
        prior = informed_prior(data_sum_stats_trunc, dt, n_lags)
    else
        raise(ArgumentError("Prior must be either 'informed' or a vector of priors given by Distributions.jl"))
    end
    return OneTimescaleModel(data, prior, data_sum_stats_trunc, epsilon, dt, T, numTrials, data_var, n_lags)
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