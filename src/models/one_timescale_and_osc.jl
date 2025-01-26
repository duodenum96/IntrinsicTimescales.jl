# src/models/one_timescale_and_osc.jl
# Model with one timescale and one oscillation

module OneTimescaleAndOsc

using Distributions
using Statistics
using ..Models
using ..OrnsteinUhlenbeck
using BayesianINT.Utils
using BayesianINT

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
One-timescale OU process model with oscillation
Parameters: [tau, freq, coeff]
Additional field: data_osc if the user wants to use combined_distance
"""
struct OneTimescaleAndOscModel <: AbstractTimescaleModel
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
end

"""
5 ways to construct a OneTimescaleAndOscModel:
1 - summary_method == :acf, fitmethod == :abc
one_timescale_and_osc_model(data, time, :abc; summary_method=:acf, prior=nothing, n_lags=nothing, 
                    distance_method=nothing, distance_combined=false, weights=nothing)

2 - summary_method == :acf, fitmethod == :optimization
one_timescale_and_osc_model(data, time, :optimization; summary_method=:acf, n_lags=nothing, 
                    optalg=nothing, distance_method=nothing, distance_combined=false, weights=nothing)

3 - summary_method == :psd, fitmethod == :abc
one_timescale_and_osc_model(data, time, :abc, summary_method == :psd, prior=nothing, 
                    distance_method=nothing, freqlims=nothing, distance_combined=false, weights=nothing)

4 - summary_method == :psd, fitmethod == :optimization
one_timescale_and_osc_model(data, time, :optimization, summary_method=:psd, optalg=nothing, 
                    distance_method=nothing, distance_combined=false, weights=nothing)

5 - summary_method == nothing, fitmethod == :acw
one_timescale_and_osc_model(data, time, :acw; summary_method=nothing, n_lags=nothing, 
                    acwtypes=[:acw0, acw50, acwe, tau, knee], dims=ndims(data))
"""
function one_timescale_and_osc_model(data, time, fit_method;
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
                                     data_mean=mean(data),
                                     data_sd=std(data),
                                     freqlims=nothing,
                                     freq_idx=nothing,
                                     dims=ndims(data),
                                     distance_combined=false,
                                     weights=[0.5, 0.5],
                                     data_tau=nothing, data_osc=nothing)

    # case 1: acf and abc
    if summary_method == :acf && fit_method == :abc
        acf = comp_ac_fft(data)
        acf_mean = mean(acf, dims=1)[:]
        lags_samples = 0.0:(size(data, dims)-1)
        if isnothing(n_lags)
            n_lags = floor(Int, acw0(lags_samples, acf_mean) * 1.5)
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
                                       optalg, acwtypes, distance_method, data_sum_stats,
                                       dt, T,
                                       numTrials, data_mean, data_sd, freqlims, n_lags,
                                       freq_idx,
                                       dims, distance_combined, weights, data_tau, data_osc)

        # case 2: acf and optimization
    elseif summary_method == :acf && fit_method == :optimization
        error("Optimization not implemented yet.")

        # case 3: psd and abc
    elseif summary_method == :psd && fit_method == :abc
        fs = 1 / dt
        psd, freqs = comp_psd(data, fs)
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
                                       prior, optalg, acwtypes, distance_method,
                                       data_sum_stats,
                                       dt, T, numTrials, data_mean, data_sd, freqlims,
                                       n_lags,
                                       freq_idx, dims, distance_combined, weights, data_tau,
                                       data_osc)

        # case 4: psd and optimization
    elseif summary_method == :psd && fit_method == :optimization
        error("Optimization not implemented yet.")

        # case 5: acw
    elseif fit_method == :acw
        possible_acwtypes = [:acw0, :acw50, :acweuler, :tau, :knee]
        acf_acwtypes = [:acw0, :acw50, :acweuler, :tau]
        n_acw = length(acwtypes)
        if n_acw == 0
            error("No ACW types specified. Possible ACW types: $(possible_acwtypes)")
        end
        result = Vector{Vector{<:Real}}(undef, n_acw)
        acwtypes = check_acwtypes(acwtypes, possible_acwtypes)
        if any(in.(acf_acwtypes, [acwtypes]))
            acf = comp_ac_fft(data; dims=dims)
            lags_samples = 0.0:(size(data, dims)-1)
            lags = lags_samples * dt
            if any(in.(:acw0, [acwtypes]))
                acw0_idx = findfirst(acwtypes .== :acw0)
                acw0_result = acw0(lags, acf; dims=dims)
                result[acw0_idx] = acw0_result
            end
            if any(in.(:acw50, [acwtypes]))
                acw50_idx = findfirst(acwtypes .== :acw50)
                acw50_result = acw50(lags, acf; dims=dims)
                result[acw50_idx] = acw50_result
            end
            if any(in.(:acweuler, [acwtypes]))
                acweuler_idx = findfirst(acwtypes .== :acweuler)
                acweuler_result = acweuler(lags, acf; dims=dims)
                result[acweuler_idx] = acweuler_result
            end
            if any(in.(:tau, [acwtypes]))
                tau_idx = findfirst(acwtypes .== :tau)
                tau_result = fit_expdecay(collect(lags), acf; dims=dims)
                result[tau_idx] = tau_result
            end
        end

        if any(in.(:knee, [acwtypes]))
            knee_idx = findfirst(acwtypes .== :knee)
            fs = 1 / dt
            psd, freqs = comp_psd(data, fs, dims=dims)
            knee_result = tau_from_knee(find_knee_frequency(psd, freqs; dims=dims))
            result[knee_idx] = knee_result
        end
        return result
    end
end

# Implementation of required methods
function Models.generate_data(model::OneTimescaleAndOscModel, theta)
    return generate_ou_with_oscillation(theta, model.dt, model.T, model.numTrials,
                                        model.data_mean, model.data_sd)
end

function Models.summary_stats(model::OneTimescaleAndOscModel, data)
    if model.summary_method == :acf
        return mean(comp_ac_fft(data; n_lags=model.n_lags), dims=1)[:][1:model.n_lags]
    elseif model.summary_method == :psd
        return mean(comp_psd(data, 1 / model.dt)[1], dims=1)[:][model.freq_idx]
    else
        throw(ArgumentError("Summary method must be :acf or :psd"))
    end
end

function combined_distance(model::OneTimescaleAndOscModel, simulation_summary, data_summary,
                           weights, distance_method,
                           data_tau, simulation_tau, data_osc, simulation_osc)
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

function Models.solve(model::OneTimescaleAndOscModel, param_dict=nothing)
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
    end
    posterior_samples = abc_record[end].theta_accepted
    posterior_MAP = find_MAP(posterior_samples, param_dict[:N])
    return posterior_samples, posterior_MAP, abc_record
end

end # module OneTimescaleAndOsc