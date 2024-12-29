# src/models/ou_process.jl

module OrnsteinUhlenbeck
using Revise
using Random
using Distributions
using BayesianINT
using ..Models
using NonlinearSolve

export OneTimescaleModel, generate_ou_process, informed_prior_one_timescale

"""
Generate an Ornstein-Uhlenbeck process with a single timescale.

# Arguments
- `tau::Float64`: Timescale
- `true_D::Float64`: Variance of data. This will be used to manually scale the OU process so that 
Bayesian inference doesn't have to deal with it. 
- `deltaT::Float64`: Time step size
- `T::Float64`: Total time length
- `num_trials::Int64`: Number of trials/trajectories to generate

# Returns
- Matrix{Float64}: Generated OU process data with dimensions (num_trials, num_timesteps)

The process is generated using the Euler-Maruyama method with the specified time step deltaT.
"""
function generate_ou_process(
    tau::Float64,
    true_D::Float64,
    deltaT::Float64,
    T::Float64,
    num_trials::Int64
)
    D_normalized = 1 / tau
    num_bin = Int(T / deltaT)
    noise = randn(num_trials, num_bin)
    ou = zeros(num_trials, num_bin)
    ou[:, 1] = noise[:, 1]

    for i in 2:num_bin
        ou[:, i] = @views ou[:, i-1] .- (ou[:, i-1] / tau) * deltaT .+
                           sqrt(2*D_normalized*deltaT) * noise[:, i-1]
    end
    # Scale the data by the true variance
    ou = ou * sqrt(true_D)
    return ou
end

function informed_prior_one_timescale(data::AbstractMatrix)
    # TODO: Implement this
    data_ac = comp_ac_fft(data; normalize=false)
    # Fit an exponential decay to the data_ac and make informed priors for tau and D
end

"""
One-timescale OU process model
"""
struct OneTimescaleModel <: AbstractTimescaleModel
    data::Matrix{Float64}
    prior::Vector{Distribution}
    data_sum_stats::Vector{Float64}
    epsilon::Float64
    deltaT::Float64
    binSize::Float64
    T::Float64
    numTrials::Int
    data_mean::Float64
    data_var::Float64
end

# Implementation of required methods

function Models.generate_data(model::OneTimescaleModel, theta)
    tau = theta
    return generate_ou_process(tau, model.data_var, model.deltaT, model.T, model.numTrials)
end

function Models.summary_stats(model::OneTimescaleModel, data)
    return comp_ac_fft(data)
end

function Models.distance_function(model::OneTimescaleModel, sum_stats, data_sum_stats)
    return linear_distance(sum_stats, data_sum_stats)
end

end # module OrnsteinUhlenbeck