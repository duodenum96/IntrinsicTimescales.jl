# src/core/abc.jl
# TODO: Overall cleanup
# TODO: There are many possible performance improvements in the basic_abc function. The core 
# challange is to eliminate if statements in the for loop. 
# TODO: Switch to StaticArrays

module ABC

using ..Models
using Statistics
import Distributions as dist
import StatsBase as sb
using ProgressMeter
using LinearAlgebra
using KernelDensity

export basic_abc, pmc_abc, effective_sample_size, weighted_covar, find_MAP, get_param_dict_abc, 
    abc_results, ABCResults

"""
    ABCResults

Container for ABC results to standardize plotting interface.

# Fields
- `theta_history::Vector{Matrix{Float64}}`: History of parameter values across iterations
- `epsilon_history::Vector{Float64}`: History of epsilon values
- `acc_rate_history::Vector{Float64}`: History of acceptance rates
- `weights_history::Vector{Vector{Float64}}`: History of weights
- `final_theta::Matrix{Float64}`: Final accepted parameter values
- `final_weights::Vector{Float64}`: Final weights
- `MAP::Vector{Float64}`: Maximum A Posteriori (MAP) estimate of parameters
"""
struct ABCResults
    theta_history::Vector{Matrix{Float64}}
    epsilon_history::Vector{Float64}
    acc_rate_history::Vector{Float64}
    weights_history::Vector{Vector{Float64}}
    final_theta::Matrix{Float64}
    final_weights::Vector{Float64}
    MAP::Vector{Float64}
end

"""
    abc_results(output_record::Vector{NamedTuple})

Construct an `ABCResults` struct from PMC-ABC output records.

# Arguments
- `output_record::Vector{NamedTuple}`: Vector of named tuples containing PMC-ABC iteration results. 
  Each tuple must contain:
    - `theta_accepted`: Accepted parameter values
    - `epsilon`: Epsilon threshold value
    - `n_accepted`: Number of accepted samples
    - `n_total`: Total number of samples
    - `weights`: Importance weights

# Returns
- `ABCResults`: Struct to contain ABC results. See the documentation for `ABCResults` for more details.
"""
function abc_results(output_record::Vector{NamedTuple})
    n_steps = length(output_record)
    

    theta_history = [output_record[i].theta_accepted for i in 1:n_steps]
    epsilon_history = [output_record[i].epsilon for i in 1:n_steps]
    acc_rate_history = [output_record[i].n_accepted/output_record[i].n_total for i in 1:n_steps]
    weights_history = [output_record[i].weights for i in 1:n_steps]
    
    final_theta = output_record[end].theta_accepted
    final_weights = output_record[end].weights

    MAP = find_MAP(final_theta)
    
    return ABCResults(
        theta_history,
        epsilon_history,
        acc_rate_history,
        weights_history,
        final_theta,
        final_weights,
        MAP
    )
end

"""
    draw_theta_pmc(model, theta_prev, weights, tau_squared; jitter::Float64=1e-5)

Draw new parameter values using the PMC proposal distribution.

# Arguments
- `model`: Model instance
- `theta_prev`: Previously accepted parameters
- `weights`: Importance weights from previous iteration
- `tau_squared`: Covariance matrix for proposal distribution
- `jitter::Float64=1e-5`: Small value added to covariance diagonal for numerical stability

# Returns
- Vector of proposed parameters
"""
function draw_theta_pmc(model, theta_prev, weights, tau_squared; jitter::Float64=1e-5)
    theta_star = theta_prev[sb.sample(collect(1:size(theta_prev, 1)),
                                                sb.pweights(weights)),
                                      :]

    # Add small diagonal term to ensure positive definiteness
    jitter_matrix = jitter * Matrix(I, size(tau_squared, 1), size(tau_squared, 2))
    stabilized_cov = tau_squared + jitter_matrix

    theta = rand(dist.MvNormal(theta_star, stabilized_cov))

    # Only sample positive values
    while sum(theta .< 0) > 0
        theta = rand(dist.MvNormal(theta_star, stabilized_cov))
    end
    return theta
end

"""
    basic_abc(model::Models.AbstractTimescaleModel; kwargs...)

Perform basic ABC rejection sampling. The algorithm stops either when `max_iter` is reached or 
when `min_accepted` samples have been accepted.

# Arguments
- `model::Models.AbstractTimescaleModel`: Model to perform inference on
- `epsilon::Float64`: Acceptance threshold for distance between simulated and observed data
- `max_iter::Integer`: Maximum number of iterations to perform
- `min_accepted::Integer`: Minimum number of accepted samples required before stopping
- `pmc_mode::Bool=false`: Whether to use PMC proposal distribution instead of prior
- `weights::Array{Float64}`: Importance weights for PMC sampling (only used if pmc_mode=true)
- `theta_prev::Array{Float64}`: Previous parameters for PMC sampling (only used if pmc_mode=true)
- `tau_squared::Array{Float64}`: Covariance matrix for PMC sampling (only used if pmc_mode=true)
- `show_progress::Bool=true`: Whether to show progress bar with acceptance count and speed

# Returns
NamedTuple containing:
- `samples::Matrix{Float64}`: Matrix (max_iter × n_params) of all proposed parameters
- `isaccepted::Vector{Bool}`: Boolean mask of accepted samples for first `n_total` iterations
- `theta_accepted::Matrix{Float64}`: Matrix (n_accepted × n_params) of accepted parameters
- `distances::Vector{Float64}`: Vector of distances for first `n_total` iterations
- `n_accepted::Int`: Number of accepted samples
- `n_total::Int`: Total number of iterations performed
- `epsilon::Float64`: Acceptance threshold used
- `weights::Vector{Float64}`: Uniform weights (ones) for accepted samples
- `tau_squared::Matrix{Float64}`: Zero matrix (n_params × n_params) for basic ABC
- `eff_sample::Int`: Effective sample size (equals n_accepted in basic ABC)

# Implementation Details
1. Draws parameters either from prior (basic mode) or PMC proposal (pmc_mode)
2. Generates synthetic data and computes distance to observed data
3. Accepts parameters if distance ≤ epsilon
4. Stops when either max_iter reached or min_accepted samples accepted
5. Returns uniform weights and zero covariance matrix in basic mode
"""
function basic_abc(model::Models.AbstractTimescaleModel;
                   epsilon::Float64,
                   max_iter::Integer,
                   min_accepted::Integer,
                   pmc_mode::Bool=false,
                   weights=Array{Float64},
                   theta_prev=Array{Float64},
                   tau_squared=Array{Float64},
                   show_progress::Bool=true)
    n_theta = length(model.prior)
    samples = zeros(max_iter, n_theta)
    isaccepted = zeros(max_iter)
    distances = zeros(max_iter)
    accepted_count = 0

    prog = show_progress ? ProgressUnknown(desc="Accepted samples:", showspeed=true) :
           nothing
    iter = 0
    @inbounds for trial_count in 1:max_iter
        # Draw from prior or proposal
        if pmc_mode
            theta = draw_theta_pmc(model, theta_prev, weights, tau_squared)
        else
            theta = Models.draw_theta(model)
        end
        d = Models.generate_data_and_reduce(model, theta)
        @inbounds samples[trial_count, :] = theta
        @inbounds distances[trial_count] = d
        iter += 1
        if d <= epsilon
            accepted_count += 1
            show_progress && next!(prog)
        end

        if accepted_count == min_accepted
            accepted_count += 1
            show_progress && finish!(prog)
            break
        end
    end # for

    isaccepted = distances[1:iter] .<= epsilon
    accepted_count = sum(isaccepted)
    theta_accepted = samples[1:iter, :][isaccepted, :]

    weights = ones(length(theta_accepted))
    tau_squared = zeros(length(theta_accepted), length(theta_accepted))
    eff_sample = length(theta_accepted)

    return (samples=samples,
            isaccepted=isaccepted,
            theta_accepted=theta_accepted,
            distances=distances[1:iter],
            n_accepted=accepted_count,
            n_total=iter,
            epsilon=epsilon,
            weights=weights,
            tau_squared=tau_squared,
            eff_sample=eff_sample)
end

"""
    pmc_abc(model::Models.AbstractTimescaleModel; kwargs...)

Perform Population Monte Carlo Approximate Bayesian Computation (PMC-ABC) inference. 

# Arguments
## Basic ABC parameters
- `model::Models.AbstractTimescaleModel`: Model to perform inference on
- `epsilon_0::Real=1.0`: Initial epsilon threshold for acceptance
- `max_iter::Integer=10000`: Maximum number of iterations per step
- `min_accepted::Integer=100`: Minimum number of accepted samples required
- `steps::Integer=10`: Maximum number of PMC steps to perform
- `sample_only::Bool=false`: If true, only perform sampling without adaptation

## Acceptance rate parameters
- `minAccRate::Float64=0.01`: Minimum acceptance rate before early stopping
- `target_acc_rate::Float64=0.01`: Target acceptance rate for epsilon adaptation
- `target_epsilon::Float64=5e-3`: Target epsilon value for early stopping

## Display parameters
- `show_progress::Bool=true`: Whether to show progress bar
- `verbose::Bool=true`: Whether to print detailed progress information

## Numerical stability parameters
- `jitter::Float64=1e-6`: Small value added to covariance matrix for stability

## Epsilon selection parameters
- `distance_max::Float64=10.0`: Maximum distance to consider valid
- `quantile_lower::Float64=25.0`: Lower quantile for epsilon adjustment
- `quantile_upper::Float64=75.0`: Upper quantile for epsilon adjustment
- `quantile_init::Float64=50.0`: Initial quantile when no acceptance rate
- `acc_rate_buffer::Float64=0.1`: Buffer around target acceptance rate

## Adaptive alpha parameters
- `alpha_max::Float64=0.9`: Maximum adaptation rate
- `alpha_min::Float64=0.1`: Minimum adaptation rate
- `acc_rate_far::Float64=2.0`: Threshold for "far from target" adjustment
- `acc_rate_close::Float64=0.2`: Threshold for "close to target" adjustment
- `alpha_far_mult::Float64=1.5`: Multiplier for alpha when far from target
- `alpha_close_mult::Float64=0.5`: Multiplier for alpha when close to target

## Early stopping parameters
- `convergence_window::Integer=3`: Number of steps to check for convergence
- `theta_rtol::Float64=1e-2`: Relative tolerance for parameter convergence
- `theta_atol::Float64=1e-3`: Absolute tolerance for parameter convergence

# Returns
`ABCResults`: A struct containing:
- `ABCResults.theta_history`: Parameter value history across iterations
- `ABCResults.epsilon_history`: Epsilon value history 
- `ABCResults.acc_rate_history`: Acceptance rate history
- `ABCResults.weight_history`: Weight history
- `ABCResults.theta_final`: Final parameter values
- `ABCResults.weights_final`: Final weights
- `ABCResults.theta_map`: MAP estimate

# Early Stopping Conditions
The algorithm stops and returns results if any of these conditions are met:
1. Acceptance rate falls below `minAccRate`
2. Parameters converge within tolerances over `convergence_window` steps
3. Epsilon falls below `target_epsilon`
4. Maximum number of `steps` reached

# Implementation Details
1. First step uses basic ABC with prior sampling
2. Subsequent steps use PMC proposal with adaptive epsilon
3. Epsilon is adjusted based on acceptance rates and distance quantiles
4. Covariance and weights are updated each step unless `sample_only=true`
5. Parameter convergence is checked using both relative and absolute tolerances
"""
function pmc_abc(model::Models.AbstractTimescaleModel;
                 # Basic ABC parameters
                 epsilon_0::Real=1.0,
                 max_iter::Integer=10000,
                 min_accepted::Integer=100,
                 steps::Integer=10,
                 sample_only::Bool=false,

                 # Acceptance rate parameters
                 minAccRate::Float64=0.01,
                 target_acc_rate::Float64=0.01,
                 target_epsilon::Float64=5e-3,

                 # Display parameters
                 show_progress::Bool=true,
                 verbose::Bool=true,

                 # Numerical stability parameters
                 jitter::Float64=1e-6,

                 # Epsilon selection parameters
                 distance_max::Float64=10.0,
                 quantile_lower::Float64=25.0,
                 quantile_upper::Float64=75.0,
                 quantile_init::Float64=50.0,
                 acc_rate_buffer::Float64=0.1,

                 # Adaptive alpha parameters
                 alpha_max::Float64=0.9,
                 alpha_min::Float64=0.1,
                 acc_rate_far::Float64=2.0,
                 acc_rate_close::Float64=0.2,
                 alpha_far_mult::Float64=1.5,
                 alpha_close_mult::Float64=0.5,
                 
                 # Early stopping parameters
                 convergence_window::Integer=3,
                 theta_rtol::Float64=1e-2,
                 theta_atol::Float64=1e-3,
                 )

    # Initialize output record structure 
    output_record = Vector{NamedTuple}(undef, steps)
    epsilon = epsilon_0
    theta_history = Vector{Matrix{Float64}}() # For early stoppin
    ndim = length(model.prior)

    for i_step in 1:steps
        verbose && println("Starting step $(i_step)")
        verbose && println("epsilon = $(epsilon)")

        if i_step == 1  # First ABC calculation
            result = basic_abc(model,
                               epsilon=epsilon,
                               max_iter=max_iter,
                               min_accepted=min_accepted,
                               pmc_mode=false,
                               show_progress=show_progress)

            # Initial epsilon selection
            epsilon = select_epsilon(result.distances[:],
                                     epsilon,
                                     target_acc_rate=target_acc_rate,
                                     distance_max=distance_max,
                                     quantile_lower=quantile_lower,
                                     quantile_upper=quantile_upper,
                                     quantile_init=quantile_init,
                                     acc_rate_buffer=acc_rate_buffer,
                                     alpha_max=alpha_max,
                                     alpha_min=alpha_min,
                                     acc_rate_far=acc_rate_far,
                                     acc_rate_close=acc_rate_close,
                                     alpha_far_mult=alpha_far_mult,
                                     alpha_close_mult=alpha_close_mult)

            theta = result.theta_accepted
            tau_squared = 2.0 * cov(theta; dims=1)
            # Add stabilization
            tau_squared += jitter * Matrix(I, size(tau_squared, 1), size(tau_squared, 2))
            weights = fill(1.0 / size(theta, 1), size(theta, 1))
            # nonnan_distances = result.distances[result.distances.<10]
            # epsilon = sb.percentile(result.distances, 25)
            eff_sample = effective_sample_size(weights)

            output_record[i_step] = (theta_accepted=theta,
                                     samples=result.samples,
                                     D_accepted=result.distances,
                                     n_accepted=result.n_accepted,
                                     n_total=result.n_total,
                                     epsilon=epsilon,
                                     weights=weights,
                                     tau_squared=tau_squared,
                                     eff_sample=eff_sample)

        else
            theta_prev = output_record[i_step-1].theta_accepted
            weights_prev = output_record[i_step-1].weights
            tau_squared = output_record[i_step-1].tau_squared

            result = basic_abc(model,
                               epsilon=epsilon,
                               max_iter=max_iter,
                               min_accepted=min_accepted,
                               pmc_mode=true,
                               weights=weights_prev,
                               theta_prev=theta_prev,
                               tau_squared=tau_squared)

            theta = result.theta_accepted
            effective_sample = effective_sample_size(weights_prev)

            if sample_only
                weights = Float64[]
                tau_squared = zeros(0, 0)
            else
                weights = calc_weights(theta_prev, theta, tau_squared,
                                       weights_prev, model.prior)
                tau_squared = 2 * weighted_covar(theta, weights)
            end

            # Adaptive epsilon selection
            current_acc_rate = result.n_accepted / result.n_total
            epsilon = select_epsilon(result.distances,
                                     epsilon,
                                     target_acc_rate=target_acc_rate,
                                     current_acc_rate=current_acc_rate,
                                     iteration=i_step,
                                     total_iterations=steps,
                                     distance_max=distance_max,
                                     quantile_lower=quantile_lower,
                                     quantile_upper=quantile_upper,
                                     quantile_init=quantile_init,
                                     acc_rate_buffer=acc_rate_buffer,
                                     alpha_max=alpha_max,
                                     alpha_min=alpha_min,
                                     acc_rate_far=acc_rate_far,
                                     acc_rate_close=acc_rate_close,
                                     alpha_far_mult=alpha_far_mult,
                                     alpha_close_mult=alpha_close_mult)

            # Pass through adaptive alpha parameters when computing alpha
            alpha = compute_adaptive_alpha(i_step,
                                           current_acc_rate,
                                           target_acc_rate,
                                           alpha_max=alpha_max,
                                           alpha_min=alpha_min,
                                           total_iterations=steps,
                                           acc_rate_far=acc_rate_far,
                                           acc_rate_close=acc_rate_close,
                                           alpha_far_mult=alpha_far_mult,
                                           alpha_close_mult=alpha_close_mult)

            output_record[i_step] = (theta_accepted=theta,
                                     D_accepted=result.distances,
                                     n_accepted=result.n_accepted,
                                     n_total=result.n_total,
                                     epsilon=epsilon,
                                     weights=weights,
                                     tau_squared=tau_squared,
                                     eff_sample=effective_sample)
        end

        current_theta = mean(output_record[i_step].theta_accepted, dims=1)


        n_accept = output_record[i_step].n_accepted
        n_tot = output_record[i_step].n_total
        accept_rate = n_accept / n_tot
        if verbose
            println("Acceptance Rate = $(accept_rate)")
            println("Current theta = $(current_theta)")
            println("--------------------")
        end

        if accept_rate < minAccRate
            println("epsilon = $(epsilon)")
            println("Acceptance Rate = $(accept_rate)")
            return abc_results(output_record[1:i_step])
        end

        push!(theta_history, current_theta)
        # Check for parameter convergence if we have enough history
        if length(theta_history) >= convergence_window
            converged = true
            
            # Check convergence for each parameter dimension
            for dim in 1:ndim
                recent_means = [mean(history[:, dim]) for history in theta_history[end-convergence_window+1:end]]
                param_range = maximum(recent_means) - minimum(recent_means)
                param_mean = mean(recent_means)
                
                # Check both relative and absolute convergence
                rel_stable = param_range/abs(param_mean) < theta_rtol
                abs_stable = param_range < theta_atol
                
                if !(rel_stable || abs_stable)
                    converged = false
                    break
                end
            end

            if converged
                println("Parameters converged after $(i_step) iterations")
                return abc_results(output_record[1:i_step])
            end
        end

        if epsilon < target_epsilon
            println("epsilon = $(epsilon)")
            println("Acceptance Rate = $(accept_rate)")
            return abc_results(output_record[1:i_step])
        end
    end
    @warn "Results did not converge. Please check your results carefully."
    return abc_results(output_record)
end

"""
    calc_weights(theta_prev, theta, tau_squared, weights, prior)

Calculate importance weights for PMC-ABC algorithm.

# Arguments
- `theta_prev::Union{Vector{Float64}, Matrix{Float64}}`: Previously accepted parameters. For multiple parameters,
   each row is a sample and each column is a parameter
- `theta::Union{Vector{Float64}, Matrix{Float64}}`: Current parameters in same format as theta_prev
- `tau_squared::Matrix{Float64}`: Covariance matrix for the proposal distribution
- `weights::Vector{Float64}`: Previous iteration's importance weights
- `prior::Union{Vector, dist.Distribution}`: Prior distribution(s). Can be single distribution or vector of distributions

# Returns
- `Vector{Float64}`: Normalized importance weights (sum to 1)
"""
function calc_weights(theta_prev::Union{Vector{Float64}, Matrix{Float64}},
                      theta::Union{Vector{Float64}, Matrix{Float64}},
                      tau_squared::Matrix{Float64},
                      weights::Vector{Float64},
                      prior::Union{Vector, dist.Distribution})

    # Convert prior to Vector{Distribution} if it's a vector
    if prior isa Vector
        prior = convert(Vector{dist.Distribution}, prior)
    end

    weights_new = similar(weights)

    if size(theta, 2) == 1 # θ is always a matrix
        # Case for single parameter vector
        norm = similar(theta)
        for (i, T) in enumerate(theta)
            for j in axes(theta_prev, 2)
                norm[j] = dist.pdf(dist.Normal(theta_prev[j, 1], sqrt(tau_squared[1, 1])),
                                   T)
            end
            weights_new[i] = dist.pdf(prior[1], T) / sum(weights .* norm)
        end

        return weights_new ./ sum(weights_new)

    else
        # Case for multiple parameter vectors
        norm = zeros(size(theta_prev, 1))
        for i in axes(theta, 1)
            prior_prob = zeros(size(theta, 2))
            for j in axes(theta, 2)
                prior_prob[j] = dist.pdf(prior[j], theta[i, j])
            end
            # Assumes independent priors
            p = prod(prior_prob)

            for j in axes(theta_prev, 1)
                norm[j] = dist.pdf(dist.MvNormal(theta_prev[j, :], tau_squared),
                                   theta[i, :])
            end

            weights_new[i] = p / sum(weights .* norm)
        end

        return weights_new ./ sum(weights_new)
    end
end

"""
    weighted_covar(x::Matrix{Float64}, w::Vector{Float64})

Calculate weighted covariance matrix.

# Arguments
- `x::Matrix{Float64}`: Matrix of observations where each row is an observation and each column is a variable
- `w::Vector{Float64}`: Vector of weights corresponding to each observation (row of x)

# Returns
- `Matrix{Float64}`: Weighted covariance matrix of size (n_variables × n_variables)

"""
function weighted_covar(x::Matrix{Float64}, w::Vector{Float64})
    # Normalize weights to ensure they sum to 1
    w_norm = w ./ sum(w)
    sumw = sum(w_norm)

    # Check if weights are valid after normalization
    if !isapprox(sumw, 1.0, rtol=1e-10)
        @warn "Weights did not sum to 1 after normalization"
        return zeros(size(x, 2), size(x, 2))  # Return fallback covariance
    end

    sum2 = sum(w_norm .^ 2)

    if ndims(x) == 1
        @assert length(x) == length(w_norm)
        xbar = sum(w_norm .* x)
        var = sum(w_norm .* (x .- xbar) .^ 2)
        return var * sumw / (sumw * sumw - sum2)
    else
        xbar = [sum(w_norm .* x[:, i]) for i in axes(x, 2)]
        covar = zeros(size(x, 2), size(x, 2))
        for k in axes(x, 2)
            for j in axes(x, 2)
                for i in axes(x, 1)
                    @inbounds covar[j, k] += (x[i, j] - xbar[j]) * (x[i, k] - xbar[k]) *
                                             w_norm[i]
                end
            end
        end
        return covar * sumw / (sumw * sumw - sum2)
    end
end


"""
    effective_sample_size(w::Vector{Float64})

Calculate effective sample size (ESS) from importance weights. 

# Arguments
- `w::Vector{Float64}`: Vector of importance sampling weights (need not be normalized)

# Returns
- `Float64`: Effective sample size

# Details
The effective sample size is always less than or equal to the actual number of samples.
It reaches its maximum (equal to sample size) when all weights are equal, and approaches
its minimum (1) when one weight dominates all others.
"""
function effective_sample_size(w::Vector{Float64})
    """
    Calculates effective sample size
    :param w: array-like importance sampleing weights
    :return: float, effective sample size
    """

    sumw = sum(w)
    sum2 = sum(w .^ 2)
    return sumw * sumw / sum2
end

"""
    select_epsilon(distances::Vector{Float64}, current_epsilon::Float64; kwargs...)

Adaptively select the epsilon threshold for ABC based on acceptance rates and distance distribution.
Uses a combination of quantile-based bounds and adaptive step sizes to adjust epsilon towards
achieving the target acceptance rate.

# Arguments
## Required Arguments
- `distances::Vector{Float64}`: Vector of distances from ABC simulations
- `current_epsilon::Float64`: Current epsilon threshold value

## Optional Keyword Arguments
### Acceptance Rate Parameters
- `target_acc_rate::Float64=0.01`: Target acceptance rate to achieve
- `current_acc_rate::Float64=0.0`: Current acceptance rate
- `acc_rate_buffer::Float64=0.1`: Allowed deviation from target acceptance rate

### Iteration Parameters
- `iteration::Integer=1`: Current iteration number
- `total_iterations::Integer=100`: Total number of iterations planned

### Distance Processing Parameters
- `distance_max::Float64=10.0`: Maximum valid distance (larger values filtered out)
- `quantile_lower::Float64=25.0`: Lower quantile for epsilon bounds
- `quantile_upper::Float64=75.0`: Upper quantile for epsilon bounds
- `quantile_init::Float64=50.0`: Initial quantile for first iteration

### Adaptive Step Size Parameters
- `alpha_max::Float64=0.9`: Maximum adaptation rate
- `alpha_min::Float64=0.1`: Minimum adaptation rate
- `acc_rate_far::Float64=2.0`: Threshold for "far from target" adjustment
- `acc_rate_close::Float64=0.2`: Threshold for "close to target" adjustment
- `alpha_far_mult::Float64=1.5`: Multiplier for alpha when far from target
- `alpha_close_mult::Float64=0.5`: Multiplier for alpha when close to target

# Returns
- `Float64`: New epsilon value

# Implementation Details
1. Filters out NaN and distances larger than distance_max
2. Computes quantile-based bounds for epsilon adjustment
3. Uses adaptive alpha value based on iteration and acceptance rate (see `compute_adaptive_alpha`)
4. For first iteration (iteration=1):
   - Returns initial quantile of valid distances
5. For subsequent iterations:
   - If acceptance rate too high: decreases epsilon by (1-alpha)
   - If acceptance rate too low: increases epsilon by (1+alpha)
   - Keeps epsilon unchanged if within buffer of target rate
6. Always constrains new epsilon between quantile bounds

# Notes
- Returns current epsilon if no valid distances are found
- Uses compute_adaptive_alpha for step size calculation
- Adjustments are proportional to distance from target acceptance rate
"""
function select_epsilon(distances::Vector{Float64},
                        current_epsilon::Float64;
                        target_acc_rate::Float64=0.01,
                        current_acc_rate::Float64=0.0,
                        iteration::Integer=1,
                        total_iterations::Integer=100,
                        distance_max::Float64=10.0,
                        quantile_lower::Float64=25.0,
                        quantile_upper::Float64=75.0,
                        quantile_init::Float64=50.0,
                        acc_rate_buffer::Float64=0.1,
                        # Add adaptive alpha parameters
                        alpha_max::Float64=0.9,
                        alpha_min::Float64=0.1,
                        acc_rate_far::Float64=2.0,
                        acc_rate_close::Float64=0.2,
                        alpha_far_mult::Float64=1.5,
                        alpha_close_mult::Float64=0.5)

    # Filter out NaN and very large distances
    valid_distances = distances[.!isnan.(distances)]
    valid_distances = valid_distances[valid_distances.<distance_max]

    if isempty(valid_distances)
        @warn "No valid distances for epsilon selection"
        return current_epsilon
    end

    # Compute quantiles
    q_lower = sb.percentile(valid_distances, quantile_lower)
    q_init = sb.percentile(valid_distances, quantile_init)
    q_upper = sb.percentile(valid_distances, quantile_upper)

    # Get adaptive alpha value with all parameters
    alpha = compute_adaptive_alpha(iteration,
                                   current_acc_rate,
                                   target_acc_rate,
                                   alpha_max=alpha_max,
                                   alpha_min=alpha_min,
                                   total_iterations=total_iterations,
                                   acc_rate_far=acc_rate_far,
                                   acc_rate_close=acc_rate_close,
                                   alpha_far_mult=alpha_far_mult,
                                   alpha_close_mult=alpha_close_mult)

    # Adaptive selection based on acceptance rate
    if iteration == 1
        new_epsilon = q_init
    else
        if current_acc_rate > 0.0
            if current_acc_rate > target_acc_rate * (1.0 + acc_rate_buffer)
                new_epsilon = max(q_lower, current_epsilon * (1.0 - alpha))
            elseif current_acc_rate < target_acc_rate * (1.0 - acc_rate_buffer)
                new_epsilon = min(q_upper, current_epsilon * (1.0 + alpha))
            else
                new_epsilon = current_epsilon
            end
        end
    end

    return new_epsilon
end


"""
    compute_adaptive_alpha(iteration::Integer, current_acc_rate::Float64, target_acc_rate::Float64; kwargs...)

Compute an adaptive step size (alpha) for epsilon adjustment in ABC, based on iteration progress
and distance from target acceptance rate.

# Arguments
## Required Arguments
- `iteration::Integer`: Current iteration number
- `current_acc_rate::Float64`: Current acceptance rate
- `target_acc_rate::Float64`: Target acceptance rate to achieve

## Optional Keyword Arguments
### Bounds Parameters
- `alpha_max::Float64=0.9`: Maximum allowed alpha value
- `alpha_min::Float64=0.1`: Minimum allowed alpha value
- `total_iterations::Integer=100`: Total number of iterations planned

### Adaptation Parameters
- `acc_rate_far::Float64=2.0`: Relative difference threshold for "far from target"
- `acc_rate_close::Float64=0.2`: Relative difference threshold for "close to target"
- `alpha_far_mult::Float64=1.5`: Multiplier for alpha when far from target
- `alpha_close_mult::Float64=0.5`: Multiplier for alpha when close to target

# Returns
- `Float64`: Adaptive alpha value between `alpha_min` and `alpha_max`

# Implementation Details
1. Computes base alpha using linear decay between max and min:
   - `base_alpha = alpha_max * (1 - progress) + alpha_min * progress`
   where `progress = iteration/total_iterations`

2. Adjusts base alpha based on relative difference from target:
   - `acc_rate_diff = |current_acc_rate - target_acc_rate|/target_acc_rate`

3. Final alpha selection:
   - If `acc_rate_diff > acc_rate_far`: More aggressive adaptation
     `alpha = min(alpha_max, base_alpha * alpha_far_mult)`
   - If `acc_rate_diff < acc_rate_close`: More conservative adaptation
     `alpha = max(alpha_min, base_alpha * alpha_close_mult)`
   - Otherwise: Use base alpha
     `alpha = base_alpha`
"""
function compute_adaptive_alpha(iteration::Integer,
                                current_acc_rate::Float64,
                                target_acc_rate::Float64;
                                alpha_max::Float64=0.9,
                                alpha_min::Float64=0.1,
                                total_iterations::Integer=100,
                                acc_rate_far::Float64=2.0,
                                acc_rate_close::Float64=0.2,
                                alpha_far_mult::Float64=1.5,
                                alpha_close_mult::Float64=0.5)

    # Base decay factor based on iteration progress
    progress = iteration / total_iterations
    base_alpha = alpha_max * (1 - progress) + alpha_min * progress

    # Adjust based on how far we are from target acceptance rate
    acc_rate_diff = abs(current_acc_rate - target_acc_rate) / target_acc_rate

    if acc_rate_diff > acc_rate_far
        # Far from target: more aggressive adaptation
        alpha = min(alpha_max, base_alpha * alpha_far_mult)
    elseif acc_rate_diff < acc_rate_close
        # Close to target: more conservative adaptation
        alpha = max(alpha_min, base_alpha * alpha_close_mult)
    else
        # Normal range: use base alpha
        alpha = base_alpha
    end

    return alpha
end


"""
    find_MAP(theta_accepted::AbstractArray{Float64}, N::Integer=10000)

Find Maximum A Posteriori (MAP) estimates with grid search.

# Arguments
- `theta_accepted::AbstractArray{Float64}`: Matrix of accepted samples from the final step of ABC
- `N::Integer=10000`: Number of random grid points to evaluate for each parameter

# Returns
- `theta_map::Vector{Float64}`: MAP estimate of the parameters
"""
function find_MAP(theta_accepted::AbstractArray{Float64}, N::Integer=10000)
    num_params = size(theta_accepted, 2)

    # Create grid of positions for each parameter
    positions = zeros(Float64, N, num_params)
    for i in 1:num_params
        param = @view theta_accepted[:, i]
        @inbounds positions[:, i] = rand(dist.Uniform(minimum(param), maximum(param)), N)
    end

    # Estimate density using KDE
    kernel = [kde(theta_accepted[:, i]) for i in 1:num_params]

    # Evaluate density at grid positions
    probs = [pdf(kernel[i], positions[:, i]) for i in 1:num_params]

    # Find position with maximum probability
    max_idx = [findmax(probs[i])[2] for i in 1:num_params]
    theta_map = [positions[max_idx[i], i] for i in 1:num_params]

    return theta_map
end

"""
    get_param_dict_abc()

Get default parameter dictionary for ABC algorithm.

# Returns
Dictionary containing default values for all ABC parameters including:

## Basic ABC Parameters
- `:epsilon_0 => 1.0`: Initial epsilon threshold
- `:max_iter => 10000`: Maximum iterations per step
- `:min_accepted => 100`: Minimum number of accepted samples
- `:steps => 30`: Maximum PMC steps
- `:sample_only => false`: If true, only perform sampling without adaptation

## Acceptance Rate Parameters
- `:minAccRate => 0.01`: Minimum acceptance rate before early stopping
- `:target_acc_rate => 0.01`: Target acceptance rate
- `:target_epsilon => 1e-4`: Target epsilon for early stopping

## Display Parameters
- `:show_progress => true`: Show progress bar
- `:verbose => true`: Print detailed progress information

## Numerical Stability Parameters
- `:jitter => 1e-6`: Small value added to covariance matrix

## Epsilon Selection Parameters
- `:distance_max => 10.0`: Maximum valid distance
- `:quantile_lower => 25.0`: Lower quantile for epsilon bounds
- `:quantile_upper => 75.0`: Upper quantile for epsilon bounds
- `:quantile_init => 50.0`: Initial quantile
- `:acc_rate_buffer => 0.1`: Allowed deviation from target rate

## Adaptive Alpha Parameters
- `:alpha_max => 0.9`: Maximum adaptation rate
- `:alpha_min => 0.1`: Minimum adaptation rate
- `:acc_rate_far => 2.0`: Threshold for "far from target"
- `:acc_rate_close => 0.2`: Threshold for "close to target"
- `:alpha_far_mult => 1.5`: Multiplier when far from target
- `:alpha_close_mult => 0.5`: Multiplier when close to target

## Early Stopping Parameters
- `:convergence_window => 5`: Steps to check for convergence
- `:theta_rtol => 1e-2`: Relative tolerance for convergence
- `:theta_atol => 1e-3`: Absolute tolerance for convergence

## MAP Estimation Parameters
- `:N => 10000`: Number of grid points for MAP estimation

# Example
```julia
params = get_param_dict_abc()
params[:epsilon_0] = 0.5  # Modify initial epsilon
params[:max_iter] = 5000  # Reduce maximum iterations
```
"""
function get_param_dict_abc()
    return Dict(:epsilon_0 => 1.0,
                :max_iter => 10000,
                :min_accepted => 100,
                :steps => 30,
                :sample_only => false,

                # Acceptance rate parameters
                :minAccRate => 0.01,
                :target_acc_rate => 0.01,
                :target_epsilon => 1e-4,

                # Display parameters
                :show_progress => true,
                :verbose => true,

                # Numerical stability parameters
                :jitter => 1e-6,

                # Epsilon selection parameters
                :distance_max => 10.0,
                :quantile_lower => 25.0,
                :quantile_upper => 75.0,
                :quantile_init => 50.0,
                :acc_rate_buffer => 0.1,

                # Adaptive alpha parameters
                :alpha_max => 0.9,
                :alpha_min => 0.1,
                :acc_rate_far => 2.0,
                :acc_rate_close => 0.2,
                :alpha_far_mult => 1.5,
                :alpha_close_mult => 0.5,

                                 
                 # Early stopping parameters
                :convergence_window => 5,
                :theta_rtol => 1e-2,
                :theta_atol => 1e-3,

                # MAP N
                :N => 10000)
end




end # module
