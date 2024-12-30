# src/models/ou_process.jl

module OrnsteinUhlenbeck
using Revise
using Random
using Distributions
using BayesianINT
using ..Models
using NonlinearSolve
import DifferentialEquations as deq

export generate_ou_process, informed_prior_one_timescale
"""
Generate an Ornstein-Uhlenbeck process with a single timescale with vanilla Julia code.

# Arguments
- `tau::Float64`: Timescale
- `true_D::Float64`: Variance of data. This will be used to manually scale the OU process so that 
Bayesian inference doesn't have to deal with it. 
- `dt::Float64`: Time step size
- `T::Float64`: Total time length
- `num_trials::Int64`: Number of trials/trajectories to generate
- `backend::String`: Backend to use. Must be 'vanilla' or 'sciml'.

# Returns
- Matrix{Float64}: Generated OU process data with dimensions (num_trials, num_timesteps)

The process is generated using the Euler-Maruyama method with the specified time step dt.
"""
function generate_ou_process(tau::Union{Float64, Vector{Float64}},
                            true_D::Float64,
                            dt::Float64,
                            T::Float64,
                            num_trials::Int64;
                            backend::String="sciml")
    if backend == "vanilla"
        return generate_ou_process_vanilla(tau, true_D, dt, T, num_trials)
    elseif backend == "sciml"
        return generate_ou_process_sciml(tau, true_D, dt, T, num_trials)
    else
        error("Invalid backend: $backend. Must be 'vanilla' or 'sciml'.")
    end
end

"""
Generate an Ornstein-Uhlenbeck process with a single timescale using DifferentialEquations.jl.
"""
function generate_ou_process_sciml(
    tau::Union{Float64, Vector{Float64}},
    true_D::Float64,
    dt::Float64,
    T::Float64,
    num_trials::Int64,
)
    f = (du, u, p, t) -> du .= -u ./ p[1]
    g = (du, u, p, t) -> du .= 1.0 # Handle the variance below
    p = (tau, true_D)
    u0 = randn(num_trials) # Quick hack instead of ensemble problem
    prob = deq.SDEProblem(f, g, u0, (0.0, T), p)
    times = dt:dt:T
    sol = deq.solve(prob; saveat=times)
    sol_matrix = reduce(hcat, sol.u)
    ou_scaled = ((sol_matrix .- mean(sol_matrix, dims=2)) ./ std(sol_matrix, dims=2)) * true_D
    return ou_scaled
end

"""
Generate an Ornstein-Uhlenbeck process with a single timescale using vanilla Julia code.
"""
function generate_ou_process_vanilla(
    tau::Union{Float64, Vector{Float64}},
    true_D::Float64,
    dt::Float64,
    T::Float64,
    num_trials::Int64
)
    num_bin = Int(T / dt)
    noise = randn(num_trials, num_bin)
    ou = zeros(num_trials, num_bin)
    ou[:, 1] = noise[:, 1]
    
    for i in 2:num_bin
        ou[:, i] = @views ou[:, i-1] .- (ou[:, i-1] / tau) * dt .+
            sqrt(dt) * noise[:, i-1]
    end
    ou_scaled = ((ou .- mean(ou, dims=2)) ./ std(ou, dims=2)) * true_D

    return ou_scaled
end

function informed_prior_one_timescale(data::AbstractMatrix)
    # TODO: Implement this
    data_ac = comp_ac_fft(data; normalize=false)
    # Fit an exponential decay to the data_ac and make informed priors for tau and D
end
end # module OrnsteinUhlenbeck