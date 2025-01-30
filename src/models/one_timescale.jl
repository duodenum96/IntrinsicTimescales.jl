# src/models/one_timescale.jl

module OneTimescale

using Distributions
using ..Models
using ..OrnsteinUhlenbeck
using INT
using ComponentArrays
using DifferentiationInterface

export one_timescale_model, OneTimescaleModel

function informed_prior(data_sum_stats::Vector{<:Real}, lags_freqs; summary_method=:acf)
    if summary_method == :acf
        tau = fit_expdecay(lags_freqs, data_sum_stats)
        return [Normal(tau, 20.0)] # Convert to ms
    elseif summary_method == :psd
        tau = tau_from_knee(find_knee_frequency(data_sum_stats, lags_freqs)[2]) # Get knee frequency from Lorentzian fit
        return [Normal(tau, 20.0)] # Convert to ms
    end
end

"""
    OneTimescaleModel <: AbstractTimescaleModel

Model for inferring a single timescale from time series data using the Ornstein-Uhlenbeck process.
We don't recommend creating this model directly.  Instead, use the `one_timescale_model` function.

# Fields
- `data::AbstractArray{<:Real}`: Input time series data
- `time::AbstractVector{<:Real}`: Time points corresponding to the data
- `fit_method::Symbol`: Fitting method (:abc, :optimization, :acw, or :advi)
- `summary_method::Symbol`: Summary statistic type (:psd or :acf)
- `lags_freqs`: Lags (for ACF) or frequencies (for PSD)
- `prior`: Prior distribution(s) for parameters
- `acwtypes`: Types of ACW analysis to perform
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
- `u0::Union{Vector{Real}, Nothing}`: Initial parameter guess
"""
struct OneTimescaleModel <: AbstractTimescaleModel
    data::AbstractArray{<:Real}
    time::AbstractVector{<:Real}
    fit_method::Symbol # can be "abc", "optimization", "acw"
    summary_method::Symbol # :psd or :acf
    lags_freqs::Union{Real, AbstractVector} # lags if summary method is acf, freqs otherwise, If the user enters an empty vector, will use defaults. 
    prior::Union{Vector{<:Distribution}, Distribution, String, Nothing} # Vector of prior distributions, single distribution, or string for "informed_prior"
    acwtypes::Union{Vector{<:Symbol}, Symbol, Nothing} # Types of ACW: ACW-50, ACW-0, ACW-euler, tau, knee frequency
    distance_method::Symbol # :linear or :logarithmic
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
    u0::Union{Vector{Real}, Nothing} # Initial guess for optimization
end

"""
    one_timescale_model(data, time, fit_method; kwargs...)

Construct a OneTimescaleModel for time series analysis.

# Arguments
- `data`: Input time series data
- `time`: Time points corresponding to the data
- `fit_method`: Fitting method to use (:abc, :acw, or :advi)

# Keyword Arguments
- `summary_method=:acf`: Summary statistic type (:psd or :acf)
- `data_sum_stats=nothing`: Pre-computed summary statistics
- `lags_freqs=nothing`: Custom lags or frequencies
- `prior=nothing`: Prior distribution(s) for parameters
- `n_lags=nothing`: Number of lags for ACF
- `acwtypes=nothing`: Types of ACW analysis
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
- `weights=[0.5, 0.5]`: Distance weights
- `data_tau=nothing`: Pre-computed timescale
- `u0=nothing`: Initial parameter guess

# Returns
- `OneTimescaleModel`: Model instance configured for specified analysis method

# Notes
Three main usage patterns:
1. ACF-based ABC/ADVI: `summary_method=:acf`, `fit_method=:abc/:advi`
2. PSD-based ABC/ADVI: `summary_method=:psd`, `fit_method=:abc/:advi`
3. ACW analysis: `fit_method=:acw`, various `acwtypes`
"""
function one_timescale_model(data, time, fit_method; summary_method=:acf,
                             data_sum_stats=nothing,
                             lags_freqs=nothing, prior=nothing, n_lags=nothing,
                             acwtypes=nothing, distance_method=nothing,
                             dt=time[2] - time[1], T=time[end], numTrials=size(data, 1),
                             data_mean=mean(data),
                             data_sd=std(data), freqlims=nothing, freq_idx=nothing,
                             dims=ndims(data), distance_combined=false,
                             weights=[0.5, 0.5], data_tau=nothing, u0=nothing)

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
        if (isnothing(prior) || prior == "informed_prior") && fit_method == :abc
            prior = informed_prior(data_sum_stats, lags_freqs;
                                   summary_method=summary_method) # TODO: We are calculating a bunch of stuff twice here :(
        end
        if isnothing(distance_method)
            distance_method = :linear
        end

        if distance_combined
            data_tau = fit_expdecay(lags_freqs, data_sum_stats)
            u0 = [data_tau]
        end


        return OneTimescaleModel(data, time, fit_method, summary_method, lags_freqs, prior,
                                 acwtypes, distance_method, data_sum_stats, dt, T,
                                 numTrials, data_mean, data_sd, freqlims, n_lags, freq_idx,
                                 dims, distance_combined, weights, data_tau, u0)
        # case 2: psd
    elseif summary_method == :psd
        fs = 1 / dt
        psd, freqs = comp_psd(data, fs)
        mean_psd = mean(psd, dims=1)
        if isnothing(freqlims)
            freqlims = (0.5 / 1000.0, 100.0 / 1000.0) # Convert to kHz (units in ms)
        end
        freq_idx = (freqs .< freqlims[2]) .&& (freqs .> freqlims[1])
        lags_freqs = freqs[freq_idx]
        data_sum_stats = mean_psd[freq_idx]
        if (isnothing(prior) || prior == "informed_prior") && fit_method == :abc
            prior = informed_prior(data_sum_stats, lags_freqs;
                                   summary_method=summary_method)
        end
        if isnothing(distance_method)
            distance_method = :logarithmic
        end

        if distance_combined
            data_tau = tau_from_knee(find_knee_frequency(data_sum_stats, lags_freqs)[2])
            u0 = [data_tau]
        end

        return OneTimescaleModel(data, time, fit_method, summary_method, lags_freqs, prior,
                                 acwtypes, distance_method, data_sum_stats, dt, T,
                                 numTrials, data_mean, data_sd, freqlims, n_lags, freq_idx,
                                 dims, distance_combined, weights, data_tau, u0)
    end
end

"""
    Models.generate_data(model::OneTimescaleModel, theta)

Generate synthetic data from the Ornstein-Uhlenbeck process with given timescale.

# Arguments
- `model::OneTimescaleModel`: Model instance containing simulation parameters
- `theta`: Vector containing single timescale parameter (τ)

# Returns
- Synthetic time series data with same dimensions as model.data

# Notes
Uses the model's stored parameters:
- `data_sd`: Standard deviation for the OU process
- `dt`: Time step
- `T`: Total time span
- `numTrials`: Number of trials/trajectories
"""
function Models.generate_data(model::OneTimescaleModel, theta)
    return generate_ou_process(theta[1], model.data_sd, model.dt, model.T, model.numTrials)
end

"""
    Models.summary_stats(model::OneTimescaleModel, data)

Compute summary statistics (ACF or PSD) from time series data.

# Arguments
- `model::OneTimescaleModel`: Model instance specifying summary statistic type
- `data`: Time series data to analyze

# Returns
For ACF (`summary_method = :acf`):
- Mean autocorrelation function up to `n_lags`

For PSD (`summary_method = :psd`):
- Mean power spectral density within specified frequency range

# Notes
- ACF is computed using FFT-based method
- PSD is computed and filtered according to model.freq_idx
- Throws ArgumentError if summary_method is invalid
"""
function Models.summary_stats(model::OneTimescaleModel, data)
    if model.summary_method == :acf
        return mean(comp_ac_fft(data; n_lags=model.n_lags), dims=1)[:]
    elseif model.summary_method == :psd
        return mean(comp_psd(data, 1 / model.dt)[1], dims=1)[:][model.freq_idx]
    else
        throw(ArgumentError("Summary method must be :acf or :psd"))
    end
end

"""
    combined_distance(model::OneTimescaleModel, simulation_summary, data_summary,
                     weights, data_tau, simulation_tau)

Compute combined distance metric between simulated and observed data.

# Arguments
- `model`: OneTimescaleModel instance
- `simulation_summary`: Summary statistics from simulation
- `data_summary`: Summary statistics from observed data
- `weights`: Weights for combining distances
- `data_tau`: Timescale from observed data
- `simulation_tau`: Timescale from simulation

# Returns
- Weighted combination of summary statistic distance and timescale distance
"""
function combined_distance(model::OneTimescaleModel, simulation_summary, data_summary,
                           weights,
                           data_tau, simulation_tau)
    if model.distance_method == :linear
        distance_1 = linear_distance(simulation_summary, data_summary)
    elseif model.distance_method == :logarithmic
        distance_1 = logarithmic_distance(simulation_summary, data_summary)
    else
        throw(ArgumentError("Distance method must be :linear or :logarithmic"))
    end
    distance_2 = linear_distance(data_tau, simulation_tau)
    return weights[1] * distance_1 + weights[2] * distance_2
end

"""
    Models.distance_function(model::OneTimescaleModel, sum_stats, data_sum_stats)

Calculate the distance between summary statistics of simulated and observed data.

# Arguments
- `model::OneTimescaleModel`: Model instance
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
function Models.distance_function(model::OneTimescaleModel, sum_stats, data_sum_stats)
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
    Models.solve(model::OneTimescaleModel, param_dict=nothing)

Perform inference using the specified fitting method.

# Arguments
- `model::OneTimescaleModel`: Model instance
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
function Models.solve(model::OneTimescaleModel, param_dict=nothing)
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
            optimizer=param_dict[:optimizer],
        )
        
        return result
    end
end

end # module OneTimescale 