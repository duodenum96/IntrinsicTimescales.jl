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

export OneTimescaleAndOscModel, informed_prior

function informed_prior(psd::Vector{Float64}, freqs::Vector{Float64})
    u0 = lorentzian_initial_guess(psd, freqs)
    amp, knee = u0
    lorentzian_psd = lorentzian(freqs, [amp, knee])
    residual_psd = psd .- lorentzian_psd

    # 3) Find oscillation peaks
    osc_peak = find_oscillation_peak(residual_psd, freqs; min_freq=freqs[1],
                                     max_freq=freqs[end])

    priors = [Normal(1 / knee, 1), Normal(osc_peak, 0.1), Uniform(0.0, 1.0)]

    return priors
end

"""
One-timescale OU process model + Additive oscillation
Theta: [tau, freq, coeff]
Prior: "informed" or a vector of distributions
data_sum_stats: [psd, freqs]
"""
struct OneTimescaleAndOscModel <: AbstractTimescaleModel
    data::Matrix{<:Real}
    prior::Vector{Any}
    data_sum_stats::Tuple{Vector{<:Real}, Vector{<:Real}}
    epsilon::Real
    dt::Real
    T::Real
    numTrials::Integer
    data_mean::Real
    data_var::Real
end

# Constructor for informed prior
function OneTimescaleAndOscModel(data,
                                 prior::String,
                                 data_sum_stats, epsilon,
                                 dt, T, numTrials,
                                 data_mean, data_var)
    if prior == "informed"
        priors = informed_prior(data_sum_stats[1], data_sum_stats[2])
    else
        priors = prior
    end
    return OneTimescaleAndOscModel(data, priors, data_sum_stats, epsilon, dt, T, numTrials,
                                   data_mean, data_var)
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
function Models.summary_stats(model::OneTimescaleAndOscModel, data)
    return comp_psd(data, 1 / model.dt)
end

"""
Modified distance function to use combined distance
"""
function Models.distance_function(model::OneTimescaleAndOscModel, sum_stats, data_sum_stats)
    if any(isnan.(sum_stats[1]))
        return 1e5
    else
        return combined_distance(sum_stats[1], data_sum_stats[1], sum_stats[2]) # sum_stats[2] is the frequencies
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