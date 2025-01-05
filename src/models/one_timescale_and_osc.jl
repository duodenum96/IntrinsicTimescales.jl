# src/models/one_timescale_and_osc.jl
# Model with one timescale and one oscillation

# src/models/one_timescale.jl

module OneTimescaleAndOsc

using Distributions
using Statistics
using ..Models
using ..OrnsteinUhlenbeck
using BayesianINT
using Infiltrator
using ..Utils

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
    prior_scales::Vector{Tuple{Float64, Float64}} # a, b - a (to unit scale: y= (x-a)/(b-a); to original scale: x = y*(b-a) + a)
end

"""
Handle the scaling using outer constructor
"""
function OneTimescaleAndOscModel(data::Matrix{Float64},
                                 prior::Union{Vector{Uniform{Float64}},
                                              Vector{Distribution}},
                                 data_sum_stats::Vector{Float64}, epsilon::Float64,
                                 dt::Float64, T::Float64, numTrials::Int,
                                 data_mean::Float64, data_var::Float64)
    # Internally rescale the priors to a unit scale
    prior = convert(Vector{Distribution}, prior)
    prior_unit = [Uniform(0.0, 1.0) for _ in prior]
    # Get scales of original priors
    prior_scales = [(prior[i].a, prior[i].b - prior[i].a) for i in 1:length(prior)]
    return OneTimescaleAndOscModel(data, prior_unit, data_sum_stats, epsilon, dt, T,
                                   numTrials,
                                   data_mean, data_var, prior_scales)
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
                         peak_weight::Float64=0.3,
                         min_freq::Float64=5.0 / 1000.0,
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
    
    # Handle case where no peak is found
    if isnan(model_peak) || isnan(data_peak)
        return 1e5
    end
    
    # Normalize peak distance relative to frequency range
    freq_range = max_freq - min_freq
    peak_dist = abs(model_peak - data_peak) / freq_range
    @infiltrate
    # Combine distances with weighting
    total_dist = (1.0 - peak_weight) * psd_dist + peak_weight * peak_dist
    
    return total_dist
end

"""
Modified summary_stats to return both PSD and frequencies
"""
function Models.summary_stats(model::OneTimescaleAndOscModel, data)
    return comp_psd(data, 1 / model.dt) # Remove DC component
end

"""
Modified distance function to use combined distance
"""
function Models.distance_function(model::OneTimescaleAndOscModel, sum_stats, data_sum_stats)
    if any(isnan.(sum_stats[1])) || any(isnan.(data_sum_stats))
        return 1e5
    else
        return combined_distance(sum_stats[1], 
                               data_sum_stats, 
                               sum_stats[2])
    end
end

function Models.rescale_theta(model::OneTimescaleAndOscModel, theta)
    return [theta[i] * model.prior_scales[i][2] + model.prior_scales[i][1]
            for i in eachindex(theta)]
end

function Models.generate_data_and_reduce(model::OneTimescaleAndOscModel, theta)
    # The scales of the parameters (tau and freq) are very different. This might cause numerical
    # problems in the ABC algorithm. To avoid this, we rescale the parameters to be on the same scale.
    # We scale the priors to a unit scale and then rescale the parameters to the original scale.
    # We also rescale the data to a unit scale.
    # theta_rescaled = Models.rescale_theta(model, theta)

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
    
    for i_step in eachindex(results)
        results[i_step].theta_accepted = Models.rescale_theta(model, results[i_step].theta_accepted)
    end
    return results
end

end # module OneTimescaleAndOsc