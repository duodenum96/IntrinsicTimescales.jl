# src/models/one_timescale_and_osc_with_missing.jl

module OneTimescaleAndOscWithMissing

using Distributions
using ..Models
using ..OrnsteinUhlenbeck
using BayesianINT

export OneTimescaleAndOscWithMissingModel

function informed_prior(psd, freqs)
    u0 = lorentzian_initial_guess(psd, freqs)
    amp, knee = u0
    lorentzian_psd = lorentzian(freqs, [amp, knee])
    residual_psd = psd .- lorentzian_psd

    osc_peak = find_oscillation_peak(residual_psd, freqs; min_freq=freqs[1],
                                     max_freq=freqs[end])

    priors = [Normal(1 / knee, 1), Normal(osc_peak, 0.1), Uniform(0.0, 1.0)]

    return priors
end

struct OneTimescaleAndOscWithMissingModel <: AbstractTimescaleModel
    data::Matrix{Float64}
    times::Vector{Float64}
    prior::Vector{Any}
    data_sum_stats::Tuple{Vector{Float64}, Vector{Float64}}
    epsilon::Float64
    dt::Float64
    T::Float64
    numTrials::Int
    data_mean::Float64
    data_var::Float64
    missing_mask::Matrix{Bool}
end

function OneTimescaleAndOscWithMissingModel(data, times, prior, data_sum_stats, epsilon, dt,
                                            T, numTrials, data_mean, data_var)
    if prior == "informed"
        prior2 = informed_prior(data_sum_stats[1], data_sum_stats[2])
    elseif prior isa Vector{Distribution}
        prior2 = prior
    else
        error("Invalid prior type")
    end
    missing_mask = isnan.(data)
    return OneTimescaleAndOscWithMissingModel(data, times, prior2, data_sum_stats, epsilon,
                                              dt, T, numTrials, data_mean, data_var,
                                              missing_mask)
end

function Models.generate_data(model::OneTimescaleAndOscWithMissingModel, theta)
    data = generate_ou_with_oscillation(theta, model.dt, model.T, model.numTrials,
                                        model.data_mean, model.data_var)
    data[model.missing_mask] .= NaN
    return data
end

"""
Compute combined distance using PSD shape and peak location
"""
function combined_distance(model_psd::Vector{Float64},
                           data_psd::Vector{Float64},
                           freqs::Vector{Float64};
                           peak_weight::Float64=0.45,
                           knee_weight::Float64=0.45,
                           psd_weight::Float64=0.1,
                           min_freq::Float64=2.0 / 1000.0,
                           max_freq::Float64=50.0 / 1000.0)
    # 1. Regular PSD distance
    psd_dist = logarithmic_distance(model_psd, data_psd)

    # 2. Peak frequency distance
    model_knee, model_peak = Utils.fooof_fit(model_psd, freqs,
                                             min_freq=min_freq,
                                             max_freq=max_freq)
    data_knee, data_peak = Utils.fooof_fit(data_psd, freqs,
                                           min_freq=min_freq,
                                           max_freq=max_freq)

    # Handle case where no peak or knee is found
    if isnan(model_peak) || isnan(data_peak) || isnan(model_knee) || isnan(data_knee)
        return 1e5
    end

    # Normalize peak distance relative to frequency range
    freq_range = max_freq - min_freq
    peak_dist = abs(model_peak - data_peak) / freq_range
    knee_dist = abs(model_knee - data_knee) / freq_range
    # Combine distances with weighting
    total_dist = psd_weight * psd_dist + peak_weight * peak_dist + knee_weight * knee_dist

    return total_dist
end

"""
Modified summary_stats to return both PSD and frequencies
"""
function Models.summary_stats(model::OneTimescaleAndOscWithMissingModel, data)
    return comp_psd_lombscargle(model.times, data, model.missing_mask, model.dt)
end

function Models.distance_function(model::OneTimescaleAndOscWithMissingModel, sum_stats, data_sum_stats)
    # Some power values can be negative, replace with some tiny number
    if any(sum_stats[1] .< 0)
        sum_stats[1][sum_stats[1] .< 0] .= 1e-10
    end
    if any(data_sum_stats[1] .< 0)
        data_sum_stats[1][data_sum_stats[1] .< 0] .= 1e-10
    end
    return combined_distance(sum_stats[1], data_sum_stats[1], sum_stats[2])
end

function Models.generate_data_and_reduce(model::OneTimescaleAndOscWithMissingModel, theta)
    synth = Models.generate_data(model, theta)
    sum_stats = Models.summary_stats(model, synth)
    d = Models.distance_function(model, sum_stats, model.data_sum_stats)
    return d
end

end