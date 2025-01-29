module Models

using Distributions

export AbstractTimescaleModel, BaseModel, check_inputs, check_acwtypes, solve

abstract type AbstractTimescaleModel end

"""
Base model interface for ABC computations
"""
struct BaseModel <: AbstractTimescaleModel
    data
    time
    data_sum_stats
    fitmethod::Symbol # can be "abc", "optimization", "acw"
    summary_method::Symbol # :psd or :acf
    lags_freqs::AbstractVector{<:Real} # :lags if summary method is acf, freqs otherwise
    prior::Union{Vector{<:Distribution}, Distribution, String} # Vector of prior distributions or string for "informed_prior"
    optalg::Symbol # Optimization algorithm for Optimization.jl
    acwtypes::Union{Vector{<:Symbol}, Symbol} # Types of ACW: ACW-50, ACW-0, ACW-e, tau, knee frequency
    distance_method::Symbol # :linear or :logarithmic
    dt::Real
    T::Real
    numTrials::Real
    data_mean::Real
    data_sd::Real
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

function rescale_theta end

function solve(model::AbstractTimescaleModel, param_dict=nothing) end

# Combined generation and reduction step
function generate_data_and_reduce(model::AbstractTimescaleModel, theta)
    synth = generate_data(model, theta)
    sum_stats = summary_stats(model, synth)
    d = distance_function(model, sum_stats, model.data_sum_stats)
    return d
end

# General method for drawing parameters from prior distributions
function draw_theta(model::AbstractTimescaleModel)
    return [rand(p) for p in model.prior]
end

# Add these implementations for BaseModel

function generate_data(model::BaseModel, theta)
    # For testing, just return the stored data
    return model.data
end

function summary_stats(model::BaseModel, data)
    # For testing, just return the stored summary stats
    return model.data_sum_stats
end

function distance_function(model::BaseModel, sum_stats1, sum_stats2)
    # Simple Euclidean distance for testing
    return sqrt(sum((sum_stats1 .- sum_stats2) .^ 2))
end

function check_inputs(fitmethod, summary_method)
    if !(fitmethod in [:abc, :optimization, :acw])
        throw(ArgumentError("fitmethod must be :abc, :optimization, or :acw"))
    end

    if !(summary_method in [:acf, :psd])
        throw(ArgumentError("summary_method must be :acf or :psd"))
    end
end

function check_acwtypes(acwtypes, possible_acwtypes)
    if acwtypes isa Symbol
        acwtypes = [acwtypes]
    end
    if !any(reduce(hcat, [acwtypes[i] .== possible_acwtypes for i in eachindex(acwtypes)]))
        error("Possible acwtypes: $(possible_acwtypes)")
    end
    return acwtypes
end

end # module
