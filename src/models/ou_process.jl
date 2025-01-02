# src/models/ou_process.jl

module OrnsteinUhlenbeck
using Revise
using Random
using Distributions
using BayesianINT
using ..Models
using NonlinearSolve
import DifferentialEquations as deq
using Infiltrator

export generate_ou_process, generate_ou_with_oscillation, informed_prior_one_timescale, generate_ou_process_sciml
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
                            duration::Float64,
                            num_trials::Int64;
                            backend::String="sciml",
                            standardize::Bool=true)
    if backend == "vanilla"
        return generate_ou_process_vanilla(tau, true_D, dt, duration, num_trials)
    elseif backend == "sciml"
        ou, sol = generate_ou_process_sciml(tau, true_D, dt, duration, num_trials, standardize)
        if sol.retcode == deq.ReturnCode.Success
            return ou
        else
            ou = NaN * ones(num_trials, Int(duration / dt))
            return ou
        end
    else
        error("Invalid backend: $backend. Must be 'vanilla' or 'sciml'.")
    end
end

"""
Generate an Ornstein-Uhlenbeck process with a single timescale using DifferentialEquations.jl.
"""
function generate_ou_process_sciml(
    tau::Union{T, Vector{T}},
    true_D::Float64,
    dt::Float64,
    duration::Float64,
    num_trials::Int64,
    standardize::Bool=true
) where T <: Real
    f = (du, u, p, t) -> du .= -u ./ p[1]
    g = (du, u, p, t) -> du .= sqrt(2.0 / p[1])
    p = [tau, true_D]
    u0 = randn(num_trials) # Quick hack instead of ensemble problem
    prob = deq.SDEProblem(f, g, u0, (0.0, duration), p)
    times = dt:dt:duration
    sol = deq.solve(prob, deq.SOSRI(); saveat=times)
    sol_matrix = reduce(hcat, sol.u)
    if standardize
        ou_scaled = ((sol_matrix .- mean(sol_matrix, dims=2)) ./ std(sol_matrix, dims=2)) * true_D
    else
        ou_scaled = sol_matrix
    end
    return ou_scaled, sol
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

"""
Generate a one-timescale OU process with an additive oscillation.

# Arguments
- `theta::Vector{Float64}`: [timescale of OU, frequency of oscillation, coefficient for OU]
- `dt::Float64`: Time step size for the OU process generation
- `bin_size::Float64`: Bin size for binning data and computing autocorrelation
- `T::Float64`: Duration of trials
- `num_trials::Int`: Number of trials
- `data_mean::Float64`: Mean value of the OU process (average of firing rate)
- `data_var::Float64`: Variance of the OU process (variance of firing rate)

# Returns
- `Tuple{Matrix{Float64}, Int}`: Tuple containing:
  - Matrix of binned spike-counts (num_trials × num_bins)
  - Number of bins/samples per trial
"""
function generate_ou_with_oscillation(theta::Vector{T},
                                      dt::Float64,
                                      duration::Float64,
                                      num_trials::Int,
                                      data_mean::Float64,
                                      data_var::Float64) where T <: Real
    # Extract parameters
    tau = theta[1]
    freq = theta[2]
    coeff = theta[3]

    # Make sure coeff is bounded between 0 and 1
    if coeff < 0.0
        coeff = 1e-4
    elseif coeff > 1.0
        coeff = 1.0 - 1e-4
    end

    # Generate OU process and oscillation
    ou = generate_ou_process_sciml(tau, data_var, dt, duration, num_trials, false)[1]

    # Create time matrix and random phases
    time_mat = repeat(collect(dt:dt:duration), 1, num_trials)'
    phases = rand(num_trials, 1) * 2π

    # Generate oscillation and combine with OU
    oscil = sqrt(2.0) * sin.(phases .+ 2π * freq * time_mat)
    data = sqrt(1.0 - coeff) * oscil .+ sqrt(coeff) * ou

    data = (data .- mean(data, dims=2)) ./ std(data, dims=2)

    # Scale to match target mean and variance
    data_scaled = data_var * data .+ data_mean

    return data_scaled
end

function informed_prior_one_timescale(data::AbstractMatrix)
    # TODO: Implement this
    data_ac = comp_ac_fft(data; normalize=false)
    # Fit an exponential decay to the data_ac and make informed priors for tau and D
end
end # module OrnsteinUhlenbeck