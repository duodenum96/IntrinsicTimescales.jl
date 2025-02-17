# src/models/two_timescale.jl
module TwoTimescaleModels

using ..Models
using Distributions
using IntrinsicTimescales

export TwoTimescaleModel

"""
Two-timescale OU process model
"""
struct TwoTimescaleModel <: AbstractTimescaleModel
    data::Matrix{Float64}
    prior::Vector{Distribution}
    data_sum_stats::Vector{Float64}
    epsilon::Float64
    dt::Float64
    T::Float64
    numTrials::Integer
    data_var::Float64
    n_lags::Integer
end

# Implementation of required methods
function Models.draw_theta(model::TwoTimescaleModel)
    return [rand(p) for p in model.prior]
end

function Models.generate_data(model::TwoTimescaleModel, theta)
    tau1, tau2, coeff = theta
    
    # Generate first OU process
    v1 = 1.0
    D1 = v1/tau1
    ou1 = generate_ou_process(tau1, D1, model.dt, model.T, model.numTrials)
    
    # Generate second OU process
    v2 = 1.0
    D2 = v2/tau2
    ou2 = generate_ou_process(tau2, D2, model.dt, model.T, model.numTrials)
    
    # Combine processes
    ou_combined = sqrt(coeff) * ou1 + sqrt(1-coeff) * ou2
    
    # Scale to match data statistics
    ou_std = sqrt(model.data_var)
    ou_combined = ou_std * ou_combined
    
    return ou_combined
end

function Models.summary_stats(model::TwoTimescaleModel, data)
    # ... implementation ...
end

function Models.distance_function(model::TwoTimescaleModel, synth_stats, data_stats)
    # ... implementation ...
end

end # module
