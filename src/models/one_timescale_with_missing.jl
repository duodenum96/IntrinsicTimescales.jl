# src/models/one_timescale.jl

module OneTimescaleWithMissing

using Distributions
using ..Models
using ..OrnsteinUhlenbeck
using BayesianINT

export OneTimescaleWithMissingModel

"""
One-timescale OU process model with missing data as NaNs
Strategy: In the generative model, we generate the data without NaNs, then replace the 
missing_mask with NaNs.
"""
struct OneTimescaleWithMissingModel <: AbstractTimescaleModel
    data::AbstractArray{<:Real}
    prior::Vector{<:Distribution}
    data_sum_stats::Vector{<:Real}
    epsilon::Real
    dt::Real
    T::Real
    numTrials::Integer
    data_var::Real
    n_lags::Integer
    missing_mask::AbstractArray{<:Bool}
end

"""
Constructor for OneTimescaleWithMissingModel
"""
function OneTimescaleWithMissingModel(data, prior, data_sum_stats, epsilon, dt, T, numTrials, data_var, n_lags)
    missing_mask = isnan.(data)
    return OneTimescaleWithMissingModel(data, prior, data_sum_stats, epsilon, dt, T, numTrials, data_var, n_lags, missing_mask)
end

# Implementation of required methods
function Models.generate_data(model::OneTimescaleWithMissingModel, theta)
    tau = theta
    data = generate_ou_process(tau, model.data_var, model.dt, model.T, model.numTrials; backend="sciml")
    data[model.missing_mask] .= NaN
    return data
end

function Models.summary_stats(model::OneTimescaleWithMissingModel, data)
    return comp_ac_time_missing(data, model.n_lags)
end

function Models.distance_function(model::OneTimescaleWithMissingModel, sum_stats, data_sum_stats)
    return linear_distance(sum_stats, data_sum_stats)
end

end # module OneTimescaleWithMissing