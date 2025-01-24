# src/models/one_timescale.jl

module OneTimescaleWithMissing
"""
Module for OneTimescaleWithMissingModel
As a strategy, we generate the data without NaNs, then replace the 
missing_mask with NaNs. For ACF calculation, we use the comp_ac_time_missing
function which is equivalent to using 
statsmodels.tsa.statstools.acf with the option missing="conservative". 
For PSD, we use Lomb-Scargle periodogram. 
"""

using Distributions
using ..Models
using ..OrnsteinUhlenbeck
using NaNStatistics
using BayesianINT

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
One-timescale OU process model with missing data
"""
struct OneTimescaleWithMissingModel <: AbstractTimescaleModel
    data::AbstractArray{<:Real}
    time::AbstractVector{<:Real}
    fit_method::Symbol # can be "abc", "optimization", "acw"
    summary_method::Symbol # :psd or :acf
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
    missing_mask::AbstractArray{Bool}
end

"""
5 ways to construct a OneTimescaleWithMissingModel:
1 - summary_method == :acf, fitmethod == :abc
one_timescale_with_missing_model(data, time, :abc; summary_method=:acf, prior=nothing, n_lags=nothing, 
                    distance_method=nothing, distance_combined=false, weights=nothing)

2 - summary_method == :acf, fitmethod == :optimization
one_timescale_with_missing_model(data, time, :optimization; summary_method=:acf, n_lags=nothing, 
                    optalg=nothing, distance_method=nothing, distance_combined=false, weights=nothing)

3 - summary_method == :psd, fitmethod == :abc
one_timescale_with_missing_model(data, time, :abc, summary_method == :psd, prior=nothing, 
                    distance_method=nothing, freqlims=nothing, distance_combined=false, weights=nothing)

4 - summary_method == :psd, fitmethod == :optimization
one_timescale_with_missing_model(data, time, :optimization, summary_method=:psd, optalg=nothing, 
                    distance_method=nothing, distance_combined=false, weights=nothing)

5 - summary_method == nothing, fitmethod == :acw
one_timescale_with_missing_model(data, time, :acw; summary_method=nothing, n_lags=nothing, 
                    acwtypes=[:acw0, acw50, acwe, tau, knee], dims=ndims(data))
"""
function one_timescale_with_missing_model(data, time, fit_method;
                                          summary_method=:acf,
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
                                          data_tau=nothing)
    missing_mask = isnan.(data)

    # case 1: acf and abc
    if summary_method == :acf && fit_method == :abc
        acf = comp_ac_time_missing(data)
        acf_mean = mean(acf, dims=1)[:]
        if isnothing(n_lags)
            n_lags = floor(Int, acw0(lags_samples, acf_mean) * 1.5)
        end
        lags_samples = 0:(n_lags-1)
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
                                            optalg, acwtypes, distance_method,
                                            data_sum_stats, dt, T,
                                            numTrials, data_mean, data_sd, freqlims, n_lags,
                                            freq_idx,
                                            dims, distance_combined, weights, data_tau,
                                            missing_mask)
        # case 2: acf and optimization
    elseif summary_method == :acf && fit_method == :optimization
        error("Not implemented yet. ")
        # case 3: psd and abc
    elseif summary_method == :psd && fit_method == :abc
        psd, freqs = comp_psd_lombscargle(times, data, missing_mask, dt)
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
                                            optalg, acwtypes, distance_method,
                                            data_sum_stats, dt, T,
                                            numTrials, data_mean, data_sd, freqlims, n_lags,
                                            freq_idx,
                                            dims, distance_combined, weights, data_tau,
                                            missing_mask)
        # case 4: psd and optimization
    elseif summary_method == :psd && fit_method == :optimization
        error("Not implemented yet. ")
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
            acf = comp_ac_time_missing(data)
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
            psd, freqs = comp_psd_lombscargle(times, data, missing_mask, dt)
            knee_result = tau_from_knee(find_knee_frequency(psd, freqs; dims=dims))
            result[knee_idx] = knee_result
        end
        return result
    end
end

# Implementation of required methods
function Models.generate_data(model::OneTimescaleWithMissingModel, theta)
    data = generate_ou_process(theta, model.data_sd, model.dt, model.T, model.numTrials;
                               backend="sciml")
    data[model.missing_mask] .= NaN
    return data
end

function Models.summary_stats(model::OneTimescaleWithMissingModel, data)
    if model.summary_method == :acf
        return mean(comp_ac_time_missing(data, n_lags=model.n_lags), dims=1)[:][1:model.n_lags]
    elseif model.summary_method == :psd
        return mean(comp_psd_lombscargle(model.times, data, model.missing_mask, model.dt)[1],
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

function Models.distance_function(model::OneTimescaleWithMissingModel, sum_stats,
                                  data_sum_stats)
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

function Models.solve(model::OneTimescaleWithMissingModel, param_dict=nothing)
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
                             show_progress=param_dict[:show_progress])

        posterior_samples = abc_record[end].theta_accepted
        posterior_MAP = find_MAP(posterior_samples, param_dict[:N])
        return posterior_samples, posterior_MAP, abc_record
    end
end

end # module