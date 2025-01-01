# src/models/one_timescale_and_osc.jl
# Model with one timescale and one oscillation

# src/models/one_timescale.jl

module OneTimescaleAndOsc

using Distributions
using Statistics
using ..Models
using ..OrnsteinUhlenbeck
using BayesianINT

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

function Models.summary_stats(model::OneTimescaleAndOscModel, data)
    return comp_psd(data, 1 / model.dt)[1] # Remove DC component
end

function Models.distance_function(model::OneTimescaleAndOscModel, sum_stats, data_sum_stats)
    if any(isnan.(sum_stats)) || any(isnan.(data_sum_stats))
        return 1e5
    else
        return logarithmic_distance(sum_stats, data_sum_stats)
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
    theta_rescaled = Models.rescale_theta(model, theta)

    synth = Models.generate_data(model, theta_rescaled)
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