# src/models/one_timescale_and_osc.jl
# Model with one timescale and one oscillation

module OneTimescaleAndOsc

using Distributions: Distribution, Normal, Uniform
using Statistics
using ..Models
using ..OrnsteinUhlenbeck
using INT.Utils
using INT
using DifferentiationInterface

export one_timescale_and_osc_model, OneTimescaleAndOscModel

function informed_prior(data_sum_stats::Vector{<:Real}, lags_freqs; summary_method=:psd)
    if summary_method == :psd
        u0 = lorentzian_initial_guess(data_sum_stats, lags_freqs)
        amp, knee = u0
        lorentzian_psd = lorentzian(lags_freqs, [amp, knee])
        residual_psd = data_sum_stats .- lorentzian_psd

        # Find oscillation peaks
        osc_peak = find_oscillation_peak(residual_psd, lags_freqs;
                                         min_freq=lags_freqs[1],
                                         max_freq=lags_freqs[end])

        return [Normal(1 / knee, 1.0), Normal(osc_peak, 0.1), Uniform(0.0, 1.0)]
    elseif summary_method == :acf
        tau = fit_expdecay(lags_freqs, data_sum_stats)
        u0 = lorentzian_initial_guess(data_sum_stats, lags_freqs)
        amp, knee = u0
        lorentzian_psd = lorentzian(lags_freqs, [amp, knee])
        residual_psd = data_sum_stats .- lorentzian_psd

        # Find oscillation peaks
        osc_peak = find_oscillation_peak(residual_psd, lags_freqs;
                                         min_freq=lags_freqs[1],
                                         max_freq=lags_freqs[end])
        
        return [Normal(tau, 20.0), Normal(osc_peak, 0.1), Uniform(0.0, 1.0)]
    end
end

"""
    OneTimescaleAndOscModel <: AbstractTimescaleModel

Model for inferring a single timescale and oscillation from time series data using the Ornstein-Uhlenbeck process.
Parameters: [tau, freq, coeff] representing timescale, oscillation frequency, and oscillation coefficient.

# Fields
- `data::AbstractArray{<:Real}`: Input time series data
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
- `data_mean::Real`: Mean of input data
- `data_sd::Real`: Standard deviation of input data
- `freqlims`: Frequency limits for PSD analysis
- `n_lags`: Number of lags for ACF
- `freq_idx`: Boolean mask for frequency selection
- `dims::Int`: Dimension along which to compute statistics
- `distance_combined::Bool`: Whether to use combined distance metric
- `weights::Vector{Real}`: Weights for combined distance
- `data_tau::Union{Real, Nothing}`: Pre-computed timescale
- `data_osc::Union{Real, Nothing}`: Pre-computed oscillation frequency
"""
struct OneTimescaleAndOscModel <: AbstractTimescaleModel
    data::AbstractArray{<:Real}
    time::AbstractVector{<:Real}
    fit_method::Symbol
    summary_method::Symbol
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
    data_osc::Union{Real, Nothing}
end

"""
    one_timescale_and_osc_model(data, time, fit_method; kwargs...)

Construct a OneTimescaleAndOscModel for time series analysis with oscillation.

# Arguments
- `data`: Input time series data
- `time`: Time points corresponding to the data
- `fit_method`: Fitting method to use (:abc or :advi)

# Keyword Arguments
- `summary_method=:psd`: Summary statistic type (:psd or :acf)
- `data_sum_stats=nothing`: Pre-computed summary statistics
- `lags_freqs=nothing`: Custom lags or frequencies
- `prior=nothing`: Prior distribution(s) for parameters
- `n_lags=nothing`: Number of lags for ACF
- `distance_method=nothing`: Distance metric type
- `dt=time[2]-time[1]`: Time step
- `T=time[end]`: Total time span
- `numTrials=size(data,1)`: Number of trials
- `data_mean=mean(data)`: Data mean
- `data_sd=std(data)`: Data standard deviation
- `freqlims=nothing`: Frequency limits for PSD
- `freq_idx=nothing`: Frequency selection mask
- `dims=ndims(data)`: Analysis dimension
- `distance_combined=false`: Use combined distance
- `weights=[0.5, 0.5]`: Distance weights for combined distance
- `data_tau=nothing`: Pre-computed timescale
- `data_osc=nothing`: Pre-computed oscillation frequency

# Returns
- `OneTimescaleAndOscModel`: Model instance configured for specified analysis method

# Notes
Four main usage patterns:
1. ACF-based ABC/ADVI: `summary_method=:acf`, `fit_method=:abc/:advi`
2. PSD-based ABC/ADVI: `summary_method=:psd`, `fit_method=:abc/:advi`
"""

function one_timescale_and_osc_model(data, time, fit_method;
                                     summary_method=:psd,
                                     data_sum_stats=nothing,
                                     lags_freqs=nothing,
                                     prior=nothing,
                                     n_lags=nothing,
                                     distance_method=nothing,
                                     dt=time[2] - time[1],
                                     T=time[end],
                                     numTrials=size(data, 1),
                                     data_mean=mean(data),
                                     data_sd=std(data),
                                     freqlims=nothing,
                                     freq_idx=nothing,
                                     dims=ndims(data),
                                     distance_combined=false,
                                     weights=[0.5, 0.5],
                                     data_tau=nothing, data_osc=nothing)
    data, dims = check_model_inputs(data, time, fit_method, summary_method, prior, distance_method)

    # case 1: acf and abc or advi
    if summary_method == :acf
        acf = comp_ac_fft(data)
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

        return OneTimescaleAndOscModel(data, time, fit_method, summary_method, lags_freqs,
                                       prior,
                                       distance_method, data_sum_stats,
                                       dt, T,
                                       numTrials, data_mean, data_sd, freqlims, n_lags,
                                       freq_idx,
                                       dims, distance_combined, weights, data_tau, data_osc)
        # case 3: psd and abc or advi
    elseif summary_method == :psd
        fs = 1 / dt
        psd, freqs = comp_psd_adfriendly(data, fs)
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
            amp, knee = find_knee_frequency(data_sum_stats, lags_freqs)
            data_tau = tau_from_knee(knee)
            residual_psd = data_sum_stats .- lorentzian(lags_freqs, [amp, knee])
            data_osc = find_oscillation_peak(residual_psd, lags_freqs;
                                             min_freq=freqlims[1],
                                             max_freq=freqlims[2])
        end

        return OneTimescaleAndOscModel(data, time, fit_method, summary_method, lags_freqs,
                                       prior, distance_method,
                                       data_sum_stats,
                                       dt, T, numTrials, data_mean, data_sd, freqlims,
                                       n_lags,
                                       freq_idx, dims, distance_combined, weights, data_tau,
                                       data_osc)

    end
end

# Implementation of required methods
"""
    Models.generate_data(model::OneTimescaleAndOscModel, theta::AbstractVector{<:Real})

Generate synthetic data from the Ornstein-Uhlenbeck process with oscillation.

# Arguments
- `model::OneTimescaleAndOscModel`: Model instance containing simulation parameters
- `theta::AbstractVector{<:Real}`: Vector containing parameters [tau, freq, coeff]

# Returns
- Synthetic time series data with same dimensions as model.data

# Notes
Uses the model's stored parameters:
- `dt`: Time step
- `T`: Total time span
- `numTrials`: Number of trials/trajectories
- `data_mean`: Mean of the process
- `data_sd`: Standard deviation of the process
"""
function Models.generate_data(model::OneTimescaleAndOscModel, theta::AbstractVector{<:Real})
    return generate_ou_with_oscillation(theta, model.dt, model.T, model.numTrials,
                                        model.data_mean, model.data_sd)
end

"""
    Models.summary_stats(model::OneTimescaleAndOscModel, data)

Compute summary statistics (ACF or PSD) from time series data.

# Arguments
- `model::OneTimescaleAndOscModel`: Model instance specifying summary statistic type
- `data`: Time series data to analyze

# Returns
For ACF (`summary_method = :acf`):
- Mean autocorrelation function up to `n_lags`

For PSD (`summary_method = :psd`):
- Mean power spectral density within specified frequency range

# Notes
- ACF is computed using FFT-based method
- PSD is computed using AD-friendly implementation
- Throws ArgumentError if summary_method is invalid
"""
function Models.summary_stats(model::OneTimescaleAndOscModel, data)
    if model.summary_method == :acf
        return mean(comp_ac_fft(data; n_lags=model.n_lags), dims=1)[:][1:model.n_lags]
    elseif model.summary_method == :psd
        return mean(comp_psd_adfriendly(data, 1 / model.dt)[1], dims=1)[:][model.freq_idx]
    else
        throw(ArgumentError("Summary method must be :acf or :psd"))
    end
end

"""
    combined_distance(model::OneTimescaleAndOscModel, simulation_summary, data_summary,
                     weights, distance_method, data_tau, simulation_tau, 
                     data_osc, simulation_osc)

Compute combined distance metric between simulated and observed data.

# Arguments
- `model`: OneTimescaleAndOscModel instance
- `simulation_summary`: Summary statistics from simulation
- `data_summary`: Summary statistics from observed data
- `weights`: Weights for combining distances
- `distance_method`: Distance metric type
- `data_tau`: Timescale from observed data
- `simulation_tau`: Timescale from simulation
- `data_osc`: Oscillation frequency from observed data
- `simulation_osc`: Oscillation frequency from simulation

# Returns
For ACF:
- Weighted combination of summary statistic distance and timescale distance

For PSD:
- Weighted combination of summary statistic distance, timescale distance, and oscillation frequency distance
"""
function combined_distance(model::OneTimescaleAndOscModel, simulation_summary, data_summary,
                         weights, distance_method, data_tau, simulation_tau, 
                         data_osc, simulation_osc)
    if model.summary_method == :acf
        if distance_method == :linear
            distance_1 = linear_distance(simulation_summary, data_summary)
        elseif distance_method == :logarithmic
            distance_1 = logarithmic_distance(simulation_summary, data_summary)
        end
        distance_2 = linear_distance(data_tau, simulation_tau)
        return weights[1] * distance_1 + weights[2] * distance_2
    elseif model.summary_method == :psd
        if distance_method == :linear
            distance_1 = linear_distance(simulation_summary, data_summary)
        elseif distance_method == :logarithmic
            distance_1 = logarithmic_distance(simulation_summary, data_summary)
        end
        distance_2 = linear_distance(data_tau, simulation_tau)
        distance_3 = linear_distance(data_osc, simulation_osc)
        return weights[1] * distance_1 + weights[2] * distance_2 + weights[3] * distance_3
    else
        throw(ArgumentError("Summary method must be :acf or :psd"))
    end
end

"""
    Models.distance_function(model::OneTimescaleAndOscModel, sum_stats, data_sum_stats)

Calculate the distance between summary statistics of simulated and observed data.

# Arguments
- `model::OneTimescaleAndOscModel`: Model instance
- `sum_stats`: Summary statistics from simulated data
- `data_sum_stats`: Summary statistics from observed data

# Returns
- Distance value based on model.distance_method (:linear or :logarithmic)
  or combined distance if model.distance_combined is true

# Notes
If distance_combined is true:
- For ACF: Combines ACF distance with fitted exponential decay timescale distance
- For PSD: Combines PSD distance with knee frequency timescale distance and oscillation frequency distance
"""
function Models.distance_function(model::OneTimescaleAndOscModel, sum_stats, data_sum_stats)
    if model.distance_combined
        if model.summary_method == :acf
            simulation_tau = fit_expdecay(model.lags_freqs, sum_stats)
            return combined_distance(model, sum_stats, data_sum_stats, model.weights,
                                   model.distance_method, model.data_tau, simulation_tau,
                                   nothing, nothing)
        elseif model.summary_method == :psd
            amp, knee = find_knee_frequency(sum_stats, model.lags_freqs)
            simulation_tau = tau_from_knee(knee)
            residual_psd = sum_stats .- lorentzian(model.lags_freqs, [amp, knee])
            simulation_osc = find_oscillation_peak(residual_psd, model.lags_freqs;
                                                 min_freq=model.freqlims[1],
                                                 max_freq=model.freqlims[2])
            return combined_distance(model, sum_stats, data_sum_stats, model.weights,
                                   model.distance_method, model.data_tau, simulation_tau,
                                   model.data_osc, simulation_osc)
        end
    elseif model.distance_method == :linear
        return linear_distance(sum_stats, data_sum_stats)
    elseif model.distance_method == :logarithmic
        return logarithmic_distance(sum_stats, data_sum_stats)
    else
        throw(ArgumentError("Distance method must be :linear or :logarithmic"))
    end
end

"""
    Models.fit(model::OneTimescaleAndOscModel, param_dict=nothing)

Perform inference using the specified fitting method.

# Arguments
- `model::OneTimescaleAndOscModel`: Model instance
- `param_dict=nothing`: Optional dictionary of algorithm parameters. If nothing, uses defaults.

# Returns
For ABC method:
- `posterior_samples`: Matrix of accepted parameter samples
- `posterior_MAP`: Maximum a posteriori estimate
- `abc_record`: Full record of ABC iterations

For ADVI method:
- `ADVIResult`: Container with samples, MAP estimates, variances, and full variational posterior

# Notes
- For ABC: Uses Population Monte Carlo ABC with adaptive epsilon selection
- For ADVI: Uses Automatic Differentiation Variational Inference via Turing.jl
- Parameter dictionary can be customized for each method (see get_param_dict_abc())
"""
function Models.fit(model::OneTimescaleAndOscModel, param_dict::Dict=Dict())
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

        if isnothing(param_dict)
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

end # module OneTimescaleAndOsc