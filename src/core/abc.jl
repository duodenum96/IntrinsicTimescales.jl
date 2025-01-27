# src/core/abc.jl
# TODO: Overall cleanup
# TODO: There are many possible performance improvements in the basic_abc function. The core 
# challange is to eliminate if statements in the for loop. 
# TODO: Switch to StaticArrays

module ABC

using Infiltrator
using ..Models
using Statistics
import Distributions as dist
import StatsBase as sb
using ProgressMeter
using LinearAlgebra
using KernelDensity

export basic_abc, pmc_abc, effective_sample_size, weighted_covar, find_MAP, get_param_dict_abc

function draw_theta_pmc(model, theta_prev, weights, tau_squared; jitter::Float64=1e-5)
    @inbounds theta_star = theta_prev[sb.sample(collect(1:length(theta_prev)),
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
Basic ABC rejection sampling algorithm
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
    pmc_abc(model::Models.AbstractTimescaleModel; epsilon_0=1.0, max_iter=10000, min_accepted=100, steps=10, sample_only=false, minAccRate=0.01, target_acc_rate=0.01)

Perform Population Monte Carlo Approximate Bayesian Computation (PMC-ABC) inference.

# Arguments
- `model::Models.AbstractTimescaleModel`: Model to perform inference on
- `epsilon_0::Float64=1.0`: Initial epsilon threshold for acceptance
- `max_iter::Int=10000`: Maximum number of iterations per step
- `min_accepted::Int=100`: Minimum number of accepted samples required
- `steps::Int=10`: Number of PMC steps to perform
- `sample_only::Bool=false`: If true, only perform sampling without adaptation
- `minAccRate::Float64=0.01`: Minimum acceptance rate before stopping
- `target_acc_rate::Float64=0.01`: Target acceptance rate for epsilon adaptation

# Returns
Vector of NamedTuples containing results for each PMC step, including:
- Accepted parameters (theta_accepted)
- Distances (D_accepted) 
- Number of accepted/total samples
- Epsilon threshold
- Sample weights
- Covariance matrix (tau_squared)
- Effective sample size
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
                 cov_scale::Float64=2.0,

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
            tau_squared = cov_scale * cov(theta; dims=1)
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

        if verbose
            n_accept = output_record[i_step].n_accepted
            n_tot = output_record[i_step].n_total
            accept_rate = n_accept / n_tot
            println("Acceptance Rate = $(accept_rate)")
            println("Current theta = $(current_theta)")
            println("--------------------")
        end

        if accept_rate < minAccRate
            println("epsilon = $(epsilon)")
            println("Acceptance Rate = $(accept_rate)")
            return output_record[1:i_step]
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
                return output_record[1:i_step]
            end
        end

        if epsilon < target_epsilon
            println("epsilon = $(epsilon)")
            println("Acceptance Rate = $(accept_rate)")
            return output_record[1:i_step]
        end
    end

    return output_record
end

"""
Calculates importance weights for PMC-ABC algorithm.

Parameters:
    theta_prev: Previous accepted parameters
    theta: Current parameters
    tau_squared: Covariance matrix for proposal distribution
    weights: Previous importance weights
    prior: Prior distribution(s)
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
    weighted_covar(x::Union{Vector{Float64}, Matrix{Float64}}, w::Vector{Float64})

Calculates weighted covariance matrix.

# Arguments
- `x`: 1 or 2 dimensional array of values
- `w`: 1 dimensional array of weights

# Returns
- Weighted covariance matrix of x or weighted variance if x is 1d
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
Adaptively selects epsilon based on acceptance rate and distance distribution

# Arguments
- `distances`: Vector of distances from ABC
- `current_epsilon`: Current epsilon value
- `target_acc_rate`: Target acceptance rate
- `current_acc_rate`: Current acceptance rate
- `iteration`: Current iteration number
- `total_iterations`: Total number of iterations
- `distance_max`: Maximum distance to consider valid (default: 10.0)
- `quantile_lower`: Lower quantile for epsilon adjustment (default: 25)
- `quantile_upper`: Upper quantile for epsilon adjustment (default: 75)
- `quantile_init`: Initial quantile when no acceptance rate (default: 50)
- `acc_rate_buffer`: Buffer around target acceptance rate (default: 0.1)
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
Compute adaptive alpha value based on iteration and convergence metrics

# Arguments
- `iteration`: Current iteration number
- `current_acc_rate`: Current acceptance rate
- `target_acc_rate`: Target acceptance rate
- `alpha_max`: Maximum alpha value (default: 0.9)
- `alpha_min`: Minimum alpha value (default: 0.1)
- `total_iterations`: Total number of iterations
- `acc_rate_far`: Threshold for "far from target" adjustment (default: 2.0)
- `acc_rate_close`: Threshold for "close to target" adjustment (default: 0.2)
- `alpha_far_mult`: Multiplier for alpha when far from target (default: 1.5)
- `alpha_close_mult`: Multiplier for alpha when close to target (default: 0.5)
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
    find_MAP(theta_accepted::Matrix{Float64}, N::Int)

Find the MAP estimates from posteriors with grid search.

# Arguments
- `theta_accepted::Matrix{Float64}`: Matrix of accepted samples from the final step of ABC
- `N::Int`: Number of samples for grid search

# Returns
- `theta_map::Vector{Float64}`: MAP estimates of the parameters
"""
function find_MAP(theta_accepted::Matrix{Float64}, N::Integer=10000)
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

function get_param_dict_abc()
    return Dict(:epsilon_0 => 1.0,
                :max_iter => 10000,
                :min_accepted => 100,
                :steps => 10,
                :sample_only => false,

                # Acceptance rate parameters
                :minAccRate => 0.01,
                :target_acc_rate => 0.01,
                :target_epsilon => 5e-3,

                # Display parameters
                :show_progress => true,
                :verbose => true,

                # Numerical stability parameters
                :jitter => 1e-6,
                :cov_scale => 2.0,

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
                :convergence_window => 3,
                :theta_rtol => 1e-2,
                :theta_atol => 1e-3,

                # MAP N
                :N => 10000)
end

end # module
