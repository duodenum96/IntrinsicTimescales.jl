module Models

using Distributions

export AbstractTimescaleModel, BaseModel, check_inputs, check_acwtypes, solve

"""
    AbstractTimescaleModel

Abstract type representing models for timescale inference.
All concrete model implementations should subtype this.
"""
abstract type AbstractTimescaleModel end

"""
    BaseModel <: AbstractTimescaleModel

Base model structure for timescale inference using various methods.

# Fields
- `data`: Input time series data
- `time`: Time points corresponding to the data
- `data_sum_stats`: Pre-computed summary statistics of the data
- `fitmethod::Symbol`: Fitting method to use. Options: `:abc`, `:optimization`, `:acw`
- `summary_method::Symbol`: Summary statistic type. Options: `:psd` (power spectral density) or `:acf` (autocorrelation)
- `lags_freqs::AbstractVector{<:Real}`: Lags (for ACF) or frequencies (for PSD) at which to compute summary statistics
- `prior`: Prior distributions for parameters. Can be Vector{Distribution}, single Distribution, or "informed_prior"
- `acwtypes::Union{Vector{Symbol}, Symbol}`: ACW analysis types (e.g., :ACW50, :ACW0, :ACWe, :tau, :knee)
- `distance_method::Symbol`: Distance metric type. Options: `:linear` or `:logarithmic`
- `dt::Real`: Time step between observations
- `T::Real`: Total time span of the data
- `numTrials::Real`: Number of trials/iterations
- `data_mean::Real`: Mean of the input data
- `data_sd::Real`: Standard deviation of the input data
"""
struct BaseModel <: AbstractTimescaleModel
    data
    time
    data_sum_stats
    fitmethod::Symbol # can be "abc", "optimization", "acw"
    summary_method::Symbol # :psd or :acf
    lags_freqs::AbstractVector{<:Real} # :lags if summary method is acf, freqs otherwise
    prior::Union{Vector{<:Distribution}, Distribution, String} # Vector of prior distributions or string for "informed_prior"
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

Draw parameter values from the model's prior distributions.

# Returns
- Array of proposed model parameters sampled from their respective priors
"""
function draw_theta end

"""
    generate_data(model::AbstractTimescaleModel, theta)

Generate synthetic data using the forward model with given parameters.

# Arguments
- `model`: Model instance
- `theta`: Array of model parameters

# Returns
- Synthetic dataset with same structure as the original data
"""
function generate_data end

"""
    summary_stats(model::AbstractTimescaleModel, data)

Compute summary statistics (PSD or ACF) from the data.

# Arguments
- `model`: Model instance
- `data`: Input data (original or synthetic)

# Returns
- Array of summary statistics computed according to model.summary_method
"""
function summary_stats end

"""
    distance_function(model::AbstractTimescaleModel, summary_stats, summary_stats_synth)

Compute distance between two sets of summary statistics.

# Arguments
- `model`: Model instance
- `summary_stats`: First set of summary statistics
- `summary_stats_synth`: Second set of summary statistics (typically from synthetic data)

# Returns
- Distance value according to model.distance_method
"""
function distance_function end

function rescale_theta end

function solve(model::AbstractTimescaleModel, param_dict=nothing) end

# Combined generation and reduction step
"""
    generate_data_and_reduce(model::AbstractTimescaleModel, theta)

Combined function to generate synthetic data and compute distance from observed data.
This is a convenience function commonly used in ABC algorithms.

# Arguments
- `model`: Model instance
- `theta`: Array of model parameters

# Returns
- Distance value between synthetic and observed summary statistics
"""
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

"""
    check_inputs(fitmethod, summary_method)

Validate the fitting method and summary statistic choices.

# Arguments
- `fitmethod`: Symbol specifying the fitting method
- `summary_method`: Symbol specifying the summary statistic type

# Throws
- `ArgumentError`: If invalid options are provided
"""
function check_inputs(fitmethod, summary_method)
    if !(fitmethod in [:abc, :optimization, :acw])
        throw(ArgumentError("fitmethod must be :abc, :optimization, or :acw"))
    end

    if !(summary_method in [:acf, :psd])
        throw(ArgumentError("summary_method must be :acf or :psd"))
    end
end

"""
    check_acwtypes(acwtypes, possible_acwtypes)

Validate the ACW analysis types against allowed options.

# Arguments
- `acwtypes`: Symbol or Vector of Symbols specifying ACW analysis types
- `possible_acwtypes`: Vector of allowed ACW analysis types

# Returns
- Validated vector of ACW types

# Throws
- `ErrorException`: If invalid ACW types are provided
"""
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
