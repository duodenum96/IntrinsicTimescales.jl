# src/models/one_timescale_and_osc.jl
# Model with one timescale and one oscillation

# src/models/one_timescale.jl

module OneTimescaleAndOsc

using Distributions
using Statistics
using ..Models
using ..OrnsteinUhlenbeck
using BayesianINT.Utils
using BayesianINT
using Infiltrator

export OneTimescaleAndOscModel

"""
One-timescale OU process model + Additive oscillation
Theta: [tau, freq, coeff]
"""
struct OneTimescaleAndOscModel <: AbstractTimescaleModel
    data::Matrix{Float64}
    prior::Union{Vector{Uniform{Float64}}, Vector{Distribution}}
    data_sum_stats::Vector{Float64}
    epsilon::Float64
    dt::Float64
    T::Float64
    numTrials::Int
    data_mean::Float64
    data_var::Float64
end

# Implementation of required methods
function Models.generate_data(model::OneTimescaleAndOscModel, theta)
    return generate_ou_with_oscillation(theta, model.dt, model.T, model.numTrials,
                                        model.data_mean, model.data_var)
end

"""
Compute combined distance using PSD shape and peak location
"""
function combined_distance(model_psd::Vector{Float64},
                           data_psd::Vector{Float64},
                           freqs::Vector{Float64};
                           peak_weight::Float64=0.4,
                           knee_weight::Float64=0.4,
                           psd_weight::Float64=0.2,
                           min_freq::Float64=2.0 / 1000.0,
                           max_freq::Float64=50.0 / 1000.0)
    # 1. Regular PSD distance
    psd_dist = logarithmic_distance(model_psd, data_psd)

    # 2. Peak frequency distance
    model_peak = Utils.find_oscillation_peak(model_psd, freqs,
                                             min_freq=min_freq,
                                             max_freq=max_freq)
    data_peak = Utils.find_oscillation_peak(data_psd, freqs,
                                            min_freq=min_freq,
                                            max_freq=max_freq)

    model_knee = Utils.find_knee_frequency(model_psd, freqs)
    data_knee = Utils.find_knee_frequency(data_psd, freqs)

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
function Models.summary_stats(model::OneTimescaleAndOscModel, data)
    return comp_psd(data, 1 / model.dt)
end

"""
Modified distance function to use combined distance
"""
function Models.distance_function(model::OneTimescaleAndOscModel, sum_stats, data_sum_stats)
    if any(isnan.(sum_stats[1])) || any(isnan.(data_sum_stats))
        return 1e5
    else
        return combined_distance(sum_stats[1], data_sum_stats, sum_stats[2]) # sum_stats[2] is the frequencies
    end
end

function Models.generate_data_and_reduce(model::OneTimescaleAndOscModel, theta)
    synth = Models.generate_data(model, theta)
    sum_stats = Models.summary_stats(model, synth)
    d = Models.distance_function(model, sum_stats, model.data_sum_stats)
    return d
end

function Models.bayesian_inference(model::OneTimescaleAndOscModel; epsilon_0=0.5,
                                   min_samples=100,
                                   steps=60,
                                   minAccRate=0.001,
                                   max_iter=500)
    results = pmc_abc(model;
                      epsilon_0=epsilon_0,
                      min_samples=min_samples,
                      steps=steps,
                      minAccRate=minAccRate,
                      max_iter=max_iter)

    return results
end

end # module OneTimescaleAndOsc