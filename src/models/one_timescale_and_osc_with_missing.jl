# src/models/one_timescale_and_osc_with_missing.jl

"""
    OneTimescaleAndOscWithMissing

Module for handling time series analysis with both oscillations and missing data.
Uses specialized methods for handling NaN values:
- For ACF: Uses comp_ac_time_missing for proper handling of gaps
- For PSD: Uses Lomb-Scargle periodogram for irregular sampling
"""
module OneTimescaleAndOscWithMissing

using Distributions
using Statistics
using ..Models
using ..OrnsteinUhlenbeck
using INT.Utils
using INT
using NaNStatistics
using DifferentiationInterface

export one_timescale_and_osc_with_missing_model, OneTimescaleAndOscWithMissingModel

function informed_prior(data_sum_stats::Vector{<:Real}, lags_freqs; summary_method=:psd)
    if summary_method == :psd
        u0 = lorentzian_initial_guess(data_sum_stats, lags_freqs)
        amp, knee = u0
        lorentzian_psd = lorentzian(lags_freqs, [amp, knee])
        residual_psd = data_sum_stats .- lorentzian_psd

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
        return [Normal(tau, 20.0), Normal(0.01, 0.05), Uniform(0.0, 1.0)]
    end
end

"""
    OneTimescaleAndOscWithMissingModel <: AbstractTimescaleModel

Model for inferring a single timescale and oscillation from time series data with missing values.
Parameters: [tau, freq, coeff] representing timescale, oscillation frequency, and oscillation coefficient.

# Fields
- `data::AbstractArray{<:Real}`: Input time series data (may contain NaN)
- `time::AbstractVector{<:Real}`: Time points corresponding to the data
- `fit_method::Symbol`: Fitting method (:abc, :optimization, :acw, or :advi)
- `summary_method::Symbol`: Summary statistic type (:psd or :acf)
- `lags_freqs`: Lags (for ACF) or frequencies (for PSD)
- `prior`: Prior distribution(s) for parameters
- `optalg`: Optimization algorithm for :optimization method
- `acwtypes`: Types of ACW analysis to perform
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
- `data_osc::Union{Real, Nothing}`: Pre-computed oscillation frequency
- `missing_mask::AbstractArray{Bool}`: Boolean mask indicating NaN positions
"""
struct OneTimescaleAndOscWithMissingModel <: AbstractTimescaleModel
    data::AbstractArray{<:Real}
    time::AbstractVector{<:Real}
    fit_method::Symbol
    summary_method::Symbol
    lags_freqs::Union{Real, AbstractVector}
    prior::Union{Vector{<:Distribution}, Distribution, String}
    optalg::Union{Symbol, Nothing}
    acwtypes::Union{Vector{<:Symbol}, Symbol, Nothing}
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
    missing_mask::AbstractArray{Bool}
end

function one_timescale_and_osc_with_missing_model(data, time, fit_method;
                                            summary_method=:psd,
                                            data_sum_stats=nothing,
                                            lags_freqs=nothing,
                                            prior=nothing,
                                            n_lags=nothing,
                                            optalg=nothing,
                                            acwtypes=nothing,
                                            distance_method=nothing,
                                            dt=time[2] - time[1],
                                            T=time[end],
                                            numTrials=size(data, 1),
                                            data_mean=nanmean(data),
                                            data_sd=nanstd(data),
                                            freqlims=nothing,
                                            freq_idx=nothing,
                                            dims=ndims(data),
                                            distance_combined=false,
                                            weights=[0.5, 0.5],
                                            data_tau=nothing, data_osc=nothing)
    missing_mask = isnan.(data)
    if summary_method == :acf
        acf = comp_ac_time_missing(data)
        acf_mean = mean(acf, dims=1)[:]
        lags_samples = 0:(size(data, dims)-1)
        
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

        return OneTimescaleAndOscWithMissingModel(data,
                                            time,
                                            fit_method,
                                            summary_method,
                                            lags_freqs,
                                            prior,
                                            optalg,
                                            acwtypes,
                                            distance_method,
                                            data_sum_stats,
                                            dt,
                                            T,
                                            numTrials,
                                            data_mean,
                                            data_sd,
                                            freqlims,
                                            n_lags,
                                            freq_idx,
                                            dims,
                                            distance_combined,
                                            weights,
                                            data_tau,
                                            data_osc,
                                            missing_mask)
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
            amp, knee = find_knee_frequency(data_sum_stats, lags_freqs)
            data_tau = tau_from_knee(knee)
            residual_psd = data_sum_stats .- lorentzian(lags_freqs, [amp, knee])
            data_osc = find_oscillation_peak(residual_psd, lags_freqs;
                                             min_freq=freqlims[1],
                                             max_freq=freqlims[2])
        end
        return OneTimescaleAndOscWithMissingModel(data,
                                            time,
                                            fit_method,
                                            summary_method,
                                            lags_freqs,
                                            prior,
                                            optalg,
                                            acwtypes,
                                            distance_method,
                                            data_sum_stats,
                                            dt,
                                            T,
                                            numTrials,
                                            data_mean,
                                            data_sd,
                                            freqlims,
                                            n_lags,
                                            freq_idx,
                                            dims,
                                            distance_combined,
                                            weights,
                                            data_tau,
                                            data_osc,
                                            missing_mask)
    end
end

"""
    Models.generate_data(model::OneTimescaleAndOscWithMissingModel, theta)

Generate synthetic data from the Ornstein-Uhlenbeck process with oscillation and apply missing data mask.

# Arguments
- `model::OneTimescaleAndOscWithMissingModel`: Model instance containing simulation parameters
- `theta`: Vector containing parameters [tau, freq, coeff]

# Returns
- Synthetic time series data with oscillations and NaN values at positions specified by model.missing_mask

# Notes
1. Generates complete OU process data with oscillation
2. Applies missing data mask from original data
3. Returns data with same missing value pattern as input
"""
function Models.generate_data(model::OneTimescaleAndOscWithMissingModel, theta)
    data = generate_ou_with_oscillation(theta, model.dt, model.T, model.numTrials,
                                        model.data_mean, model.data_sd)
    data[model.missing_mask] .= NaN
    return data
end

"""
    combined_distance(model::OneTimescaleAndOscWithMissingModel, simulation_summary, data_summary,
                     weights, distance_method, data_tau, simulation_tau, data_osc, simulation_osc)

Compute combined distance metric between simulated and observed data.

# Arguments
- `model`: OneTimescaleAndOscWithMissingModel instance
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
- Weighted combination of ACF distance and timescale distance

For PSD:
- Weighted combination of PSD distance, timescale distance, and oscillation frequency distance
"""
function combined_distance(model::OneTimescaleAndOscWithMissingModel, simulation_summary, data_summary,
                         weights, distance_method, data_tau, simulation_tau, data_osc, simulation_osc)
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
    Models.summary_stats(model::OneTimescaleAndOscWithMissingModel, data)

Compute summary statistics (ACF or PSD) from time series data with missing values.

# Arguments
- `model::OneTimescaleAndOscWithMissingModel`: Model instance specifying summary statistic type
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
function Models.summary_stats(model::OneTimescaleAndOscWithMissingModel, data)
    if model.summary_method == :acf
        return mean(comp_ac_time_missing(data; n_lags=model.n_lags), dims=1)[:]
    elseif model.summary_method == :psd
        return mean(comp_psd_lombscargle(model.time, data, model.missing_mask, model.dt)[1],
                    dims=1)[:][model.freq_idx]
    else
        throw(ArgumentError("Summary method must be :acf or :psd"))
    end
end

function Models.distance_function(model::OneTimescaleAndOscWithMissingModel, sum_stats, data_sum_stats)
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
    Models.solve(model::OneTimescaleAndOscWithMissingModel, param_dict=nothing)

Perform inference using the specified fitting method.

# Arguments
- `model::OneTimescaleAndOscWithMissingModel`: Model instance
- `param_dict=nothing`: Optional dictionary of algorithm parameters. If nothing, uses defaults.

# Returns
For ABC method:
- `posterior_samples`: Matrix of accepted parameter samples
- `posterior_MAP`: Maximum a posteriori estimate
- `abc_record`: Full record of ABC iterations

For ADVI method:
- `TuringResult`: Container with samples, MAP estimates, variances, and full chain

# Notes
- For ABC: Uses Population Monte Carlo ABC with adaptive epsilon selection
- For ADVI: Uses Automatic Differentiation Variational Inference via Turing.jl
- Parameter dictionary can be customized for each method (see get_param_dict_abc())
"""
function Models.solve(model::OneTimescaleAndOscWithMissingModel, param_dict=nothing)
    if model.fit_method == :abc
        if isnothing(param_dict)
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
    posterior_samples = abc_record[end].theta_accepted
    posterior_MAP = find_MAP(posterior_samples, param_dict[:N])
    return posterior_samples, posterior_MAP, abc_record
    elseif model.fit_method == :advi
        if isnothing(param_dict)
            param_dict = Dict(
                :n_samples => 4000,
                :n_iterations => 10,
                :n_elbo_samples => 20,
                :optimizer => AutoForwardDiff()
            )
        end
        
        result = fit_vi(model; 
            n_samples=param_dict[:n_samples],
            n_iterations=param_dict[:n_iterations],
            n_elbo_samples=param_dict[:n_elbo_samples],
            optimizer=param_dict[:optimizer]
        )
        
        return result
    end
end

end