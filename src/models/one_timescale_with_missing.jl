# src/models/one_timescale.jl

"""
    OneTimescaleWithMissing

Module for handling time series analysis with missing data.
Uses specialized methods for handling NaN values:
- For ACF: Uses comp_ac_time_missing (equivalent to statsmodels.tsa.statstools.acf with missing="conservative")
- For PSD: Uses Lomb-Scargle periodogram to handle irregular sampling
"""
module OneTimescaleWithMissing

using Distributions: Distribution, Normal, Uniform
using Statistics: mean, std
using ..Models
using ..OrnsteinUhlenbeck
using NaNStatistics
using INT
using DifferentiationInterface

export OneTimescaleWithMissingModel, one_timescale_with_missing_model

function informed_prior(data_sum_stats::Vector{<:Real}, lags_freqs; summary_method=:acf)
    if summary_method == :acf
        tau = fit_expdecay(lags_freqs, data_sum_stats)
        return [Normal(tau, 20.0)]
    elseif summary_method == :psd
        tau = tau_from_knee(find_knee_frequency(data_sum_stats, lags_freqs)[2]) # Get knee frequency from Lorentzian fit
        return [Normal(tau, 20.0)]
    end
end

"""
    OneTimescaleWithMissingModel <: AbstractTimescaleModel

Model for inferring a single timescale from time series data with missing values.

# Fields
- `data::AbstractArray{<:Real}`: Input time series data (may contain NaN)
- `time::AbstractVector{<:Real}`: Time points corresponding to the data
- `fit_method::Symbol`: Fitting method (:abc or :advi)
- `summary_method::Symbol`: Summary statistic type (:psd or :acf)
- `lags_freqs`: Lags (for ACF) or frequencies (for PSD)
- `prior`: Prior distribution(s) for parameters
- `distance_method::Symbol`: Distance metric type (:linear or :logarithmic)
- `data_sum_stats`: Pre-computed summary statistics
- `dt::Real`: Time step between observations
- `T::Real`: Total time span
- `numTrials::Real`: Number of trials/iterations
- `data_mean::Real`: Mean of input data (excluding NaN)
- `data_sd::Real`: Standard deviation of input data (excluding NaN)
- `freqlims`: Frequency limits for PSD analysis
- `n_lags`: Number of lags for ACF
- `freq_idx`: Boolean mask for frequency selection
- `dims::Int`: Dimension along which to compute statistics
- `distance_combined::Bool`: Whether to use combined distance metric
- `weights::Vector{Real}`: Weights for combined distance
- `data_tau::Union{Real, Nothing}`: Pre-computed timescale
- `missing_mask::AbstractArray{Bool}`: Boolean mask indicating NaN positions
"""
struct OneTimescaleWithMissingModel <: AbstractTimescaleModel
    data::AbstractArray{<:Real}
    time::AbstractVector{<:Real}
    fit_method::Symbol # can be :abc or :advi
    summary_method::Symbol # :psd or :acf
    lags_freqs::Union{Real, AbstractVector}
    prior::Union{Vector{<:Distribution}, Distribution, String}
    distance_method::Symbol
    data_sum_stats::AbstractArray{<:Real}
    dt::Real
    T::Real
    numTrials::Real
    data_mean::Real
    data_sd::Real
    freqlims::Union{Tuple{Real, Real}, Nothing}
    n_lags::Union{Int, Nothing}
    freq_idx::Union{Vector{Bool}, Nothing}
    dims::Int
    distance_combined::Bool
    weights::Vector{Real}
    data_tau::Union{Real, Nothing}
    missing_mask::AbstractArray{Bool}
end

"""
    one_timescale_with_missing_model(data, time, fit_method; kwargs...)

Construct a OneTimescaleWithMissingModel for time series analysis with missing data.

# Arguments
- `data`: Input time series data (may contain NaN)
- `time`: Time points corresponding to the data
- `fit_method`: Fitting method to use (:abc or :advi)

# Keyword Arguments
- `summary_method=:acf`: Summary statistic type (:psd or :acf)
- `data_sum_stats=nothing`: Pre-computed summary statistics
- `lags_freqs=nothing`: Custom lags or frequencies
- `prior=nothing`: Prior distribution(s) for parameters
- `n_lags=nothing`: Number of lags for ACF
- `distance_method=nothing`: Distance metric type
- `dt=time[2]-time[1]`: Time step
- `T=time[end]`: Total time span
- `numTrials=size(data,1)`: Number of trials
- `data_mean=nanmean(data)`: Data mean (excluding NaN)
- `data_sd=nanstd(data)`: Data standard deviation (excluding NaN)
- `freqlims=nothing`: Frequency limits for PSD
- `freq_idx=nothing`: Frequency selection mask
- `dims=ndims(data)`: Analysis dimension
- `distance_combined=false`: Use combined distance
- `weights=[0.5, 0.5]`: Distance weights
- `data_tau=nothing`: Pre-computed timescale

# Returns
- `OneTimescaleWithMissingModel`: Model instance configured for specified analysis method

# Notes
Four main usage patterns:
1. ACF-based ABC/ADVI: `summary_method=:acf`, `fit_method=:abc/:advi`
2. PSD-based ABC/ADVI: `summary_method=:psd`, `fit_method=:abc/:advi`
"""
function one_timescale_with_missing_model(data, time, fit_method;
                                          summary_method=:acf,
                                          data_sum_stats=nothing,
                                          lags_freqs=nothing,
                                          prior=nothing,
                                          n_lags=nothing,
                                          distance_method=nothing,
                                          dt=time[2] - time[1],
                                          T=time[end],
                                          data_mean=nanmean(data),
                                          data_sd=nanstd(data),
                                          freqlims=nothing,
                                          freq_idx=nothing,
                                          dims=ndims(data),
                                          numTrials=size(data, setdiff([1, 2], dims)[1]), 
                                          distance_combined=false,
                                          weights=[0.5, 0.5],
                                          data_tau=nothing)
    data, dims = check_model_inputs(data, time, fit_method, summary_method, prior, distance_method)
    missing_mask = isnan.(data)

    # case 1: acf and abc or advi
    if summary_method == :acf
        acf = comp_ac_time_missing(data)
        acf_mean = mean(acf, dims=1)[:]
        lags_samples = 0.0:(size(data, dims)-1)
        if isnothing(n_lags)
            n_lags = floor(Int, acw0(lags_samples, acf_mean) * 1.1)
        end
        lags_freqs = collect(lags_samples * dt)[1:n_lags]
        data_sum_stats = acf_mean[1:n_lags]

        if isnothing(prior) || prior == "informed_prior"
            prior = informed_prior(data_sum_stats, lags_freqs;
                                   summary_method=summary_method)
        end

        if isnothing(distance_method)
            distance_method = :linear
        end

        if distance_combined
            data_tau = fit_expdecay(lags_freqs, data_sum_stats)
        end

        return OneTimescaleWithMissingModel(data, time, fit_method, summary_method,
                                            lags_freqs, prior,
                                            distance_method,
                                            data_sum_stats, dt, T,
                                            numTrials, data_mean, data_sd, freqlims, n_lags,
                                            freq_idx,
                                            dims, distance_combined, weights, data_tau,
                                            missing_mask)
    # case 2: psd and abc or advi
    elseif summary_method == :psd
        psd, freqs = comp_psd_lombscargle(time, data, missing_mask, dt)
        mean_psd = mean(psd, dims=1)
        if isnothing(freqlims)
            freqlims = (0.5 / 1000.0, 100.0 / 1000.0) # Convert to kHz (units in ms)
        end
        freq_idx = (freqs .< freqlims[2]) .&& (freqs .> freqlims[1])
        lags_freqs = freqs[freq_idx]
        data_sum_stats = mean_psd[freq_idx]

        if isnothing(prior) || prior == "informed_prior"
            prior = informed_prior(data_sum_stats, lags_freqs;
                                   summary_method=summary_method)
        end

        if isnothing(distance_method)
            distance_method = :logarithmic
        end

        if distance_combined
            data_tau = tau_from_knee(find_knee_frequency(data_sum_stats, lags_freqs)[2])
        end

        return OneTimescaleWithMissingModel(data, time, fit_method, summary_method,
                                            lags_freqs, prior,
                                            distance_method,
                                            data_sum_stats, dt, T,
                                            numTrials, data_mean, data_sd, freqlims, n_lags,
                                            freq_idx,
                                            dims, distance_combined, weights, data_tau,

                                            missing_mask)
    end
end

# Implementation of required methods
"""
    Models.generate_data(model::OneTimescaleWithMissingModel, theta)

Generate synthetic data from the Ornstein-Uhlenbeck process and apply missing data mask.

# Arguments
- `model::OneTimescaleWithMissingModel`: Model instance containing simulation parameters
- `theta`: Vector containing single timescale parameter (τ)

# Returns
- Synthetic time series data with NaN values at positions specified by model.missing_mask

# Notes
1. Generates complete OU process data
2. Applies missing data mask from original data
3. Returns data with same missing value pattern as input
"""
function Models.generate_data(model::OneTimescaleWithMissingModel, theta)
    data = generate_ou_process(theta[1], model.data_sd, model.dt, model.T, model.numTrials)
    data[model.missing_mask] .= NaN
    return data
end

"""
    Models.summary_stats(model::OneTimescaleWithMissingModel, data)

Compute summary statistics (ACF or PSD) from time series data with missing values.

# Arguments
- `model::OneTimescaleWithMissingModel`: Model instance specifying summary statistic type
- `data`: Time series data to analyze (may contain NaN)

# Returns
For ACF (`summary_method = :acf`):
- Mean autocorrelation function up to `n_lags`, computed with missing data handling

For PSD (`summary_method = :psd`):
- Mean Lomb-Scargle periodogram within specified frequency range

# Notes
- ACF uses comp_ac_time_missing for proper handling of NaN values
- PSD uses Lomb-Scargle periodogram for irregular sampling
"""
function Models.summary_stats(model::OneTimescaleWithMissingModel, data)
    if model.summary_method == :acf
        return mean(comp_ac_time_missing(data, n_lags=model.n_lags), dims=1)[:]
    elseif model.summary_method == :psd
        return mean(comp_psd_lombscargle(model.time, data, model.missing_mask, model.dt)[1],
                    dims=1)[:][model.freq_idx]
    else
        throw(ArgumentError("Summary method must be :acf or :psd"))
    end
end

function combined_distance(model::OneTimescaleWithMissingModel, simulation_summary,
                           data_summary,
                           weights,
                           data_tau, simulation_tau)
    if model.distance_method == :linear
        distance_1 = linear_distance(simulation_summary, data_summary)
    elseif model.distance_method == :logarithmic
        distance_1 = logarithmic_distance(simulation_summary, data_summary)
    end
    distance_2 = linear_distance(data_tau, simulation_tau)
    return weights[1] * distance_1 + weights[2] * distance_2
end

"""
    Models.distance_function(model::OneTimescaleWithMissingModel, sum_stats, data_sum_stats)

Calculate the distance between summary statistics of simulated and observed data.

# Arguments
- `model::OneTimescaleWithMissingModel`: Model instance
- `sum_stats`: Summary statistics from simulated data
- `data_sum_stats`: Summary statistics from observed data

# Returns
- Distance value based on model.distance_method (:linear or :logarithmic)
  or combined distance if model.distance_combined is true

# Notes
If distance_combined is true:
- For ACF: Combines ACF distance with fitted exponential decay timescale distance
- For PSD: Combines PSD distance with knee frequency timescale distance
"""
function Models.distance_function(model::OneTimescaleWithMissingModel, sum_stats, data_sum_stats)
    if model.distance_combined
        if model.summary_method == :acf
            simulation_tau = fit_expdecay(model.lags_freqs, sum_stats)
        elseif model.summary_method == :psd
            simulation_tau = tau_from_knee(find_knee_frequency(sum_stats, model.lags_freqs)[2])
        end
        return combined_distance(model, sum_stats, data_sum_stats, model.weights,
                                 model.data_tau, simulation_tau)
    elseif model.distance_method == :linear
        return linear_distance(sum_stats, data_sum_stats)
    elseif model.distance_method == :logarithmic
        return logarithmic_distance(sum_stats, data_sum_stats)
    else
        throw(ArgumentError("Distance method must be :linear or :logarithmic"))
    end
end

"""
    Models.fit(model::OneTimescaleWithMissingModel, param_dict=nothing)

Perform inference using the specified fitting method.

# Arguments
- `model::OneTimescaleWithMissingModel`: Model instance
- `param_dict=nothing`: Optional dictionary of algorithm parameters. If nothing, uses defaults.

# Returns
For ABC method:
- `posterior_samples`: Matrix of accepted parameter samples
- `posterior_MAP`: Maximum a posteriori estimate
- `abc_record`: Full record of ABC iterations

For ADVI method:
- `ADVIResult`: Container with samples, MAP estimates, variances, and full chain

# Notes
- For ABC: Uses Population Monte Carlo ABC with adaptive epsilon selection
- For ADVI: Uses Automatic Differentiation Variational Inference via Turing.jl
- Parameter dictionary can be customized for each method (see get_param_dict_abc())
"""
function Models.fit(model::OneTimescaleWithMissingModel, param_dict::Dict=Dict())
    if model.fit_method == :abc
        if isempty(param_dict)
            param_dict = get_param_dict_abc()

        end

        abc_record = pmc_abc(model;
                             epsilon_0=param_dict[:epsilon_0],
                             max_iter=param_dict[:max_iter],
                             min_accepted=param_dict[:min_accepted],
                             steps=param_dict[:steps],
                             sample_only=param_dict[:sample_only],
                             minAccRate=param_dict[:minAccRate],
                             target_acc_rate=param_dict[:target_acc_rate],
                             target_epsilon=param_dict[:target_epsilon],
                             show_progress=param_dict[:show_progress],
                             verbose=param_dict[:verbose],
                             jitter=param_dict[:jitter],
                             cov_scale=param_dict[:cov_scale],
                             distance_max=param_dict[:distance_max],
                             quantile_lower=param_dict[:quantile_lower],
                             quantile_upper=param_dict[:quantile_upper],
                             quantile_init=param_dict[:quantile_init],
                             acc_rate_buffer=param_dict[:acc_rate_buffer],
                             alpha_max=param_dict[:alpha_max],
                             alpha_min=param_dict[:alpha_min],
                             acc_rate_far=param_dict[:acc_rate_far],
                             acc_rate_close=param_dict[:acc_rate_close],
                             alpha_far_mult=param_dict[:alpha_far_mult],
                             alpha_close_mult=param_dict[:alpha_close_mult],
                             convergence_window=param_dict[:convergence_window],
                             theta_rtol=param_dict[:theta_rtol],
                             theta_atol=param_dict[:theta_atol])

        return abc_record


    elseif model.fit_method == :advi
        if isempty(param_dict)
            param_dict = get_param_dict_advi()
        end
        


        result = fit_vi(model; 
            n_samples=param_dict[:n_samples],
            n_iterations=param_dict[:n_iterations],
            n_elbo_samples=param_dict[:n_elbo_samples],
            optimizer=param_dict[:autodiff]
        )
        
        return result
    end
end


end # module