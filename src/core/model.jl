module Models

using Distributions

export AbstractTimescaleModel, BaseModel

abstract type AbstractTimescaleModel end

"""
Base model interface for ABC computations
"""
struct BaseModel{T,D,P} <: AbstractTimescaleModel
    data::D
    prior::P
    data_sum_stats::T
    epsilon::Float64
end

# Required methods that need to be implemented for each model
"""
    draw_theta(model::AbstractTimescaleModel)

Draw parameters from prior distributions.
Should return an array-like iterable of proposed model parameters.
"""
function draw_theta end

"""
    generate_data(model::AbstractTimescaleModel, theta)

Generate synthetic data sets from forward model.
Should return an array/matrix/table of simulated data.
"""
function generate_data end

"""
    summary_stats(model::AbstractTimescaleModel, data)

Compute summary statistics from data.
Should return an array-like iterable of summary statistics.
"""
function summary_stats end

"""
    distance_function(model::AbstractTimescaleModel, summary_stats, summary_stats_synth)

Compute distance between summary statistics.
Should return a distance D for comparing to the acceptance tolerance (epsilon).
"""
function distance_function end

# Combined generation and reduction step
function generate_data_and_reduce(model::AbstractTimescaleModel, theta)
    synth = generate_data(model, theta)
    sum_stats = summary_stats(model, synth)
    d = distance_function(model, sum_stats, model.data_sum_stats)
    return d
end

end # module
