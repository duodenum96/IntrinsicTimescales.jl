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

export basic_abc, pmc_abc, effective_sample_size, weighted_covar, find_MAP

function draw_theta_pmc(model, theta_prev, weights, tau_squared)
    @inbounds theta_star = theta_prev[sb.sample(collect(1:length(theta_prev)), sb.pweights(weights)),
                            :]

    # Add small diagonal term to ensure positive definiteness
    jitter = 1e-5 * Matrix(I, size(tau_squared, 1), size(tau_squared, 2))
    stabilized_cov = tau_squared + jitter

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
                   tau_squared=Array{Float64})
    n_theta = length(model.prior)
    samples = zeros(max_iter, n_theta)
    isaccepted = zeros(max_iter)
    distances = zeros(max_iter)
    accepted_count = 0

    prog = ProgressUnknown(desc="Accepted samples:", showspeed=true) # Progress meter for accepted samples
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
            next!(prog)
        end
        if accepted_count == min_accepted
            accepted_count += 1
            finish!(prog)
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
                 epsilon_0::Real=1.0,
                 max_iter::Integer=10000,
                 min_accepted::Integer=100,
                 steps::Integer=10,
                 sample_only::Bool=false,
                 minAccRate::Float64=0.01,
                 target_acc_rate::Float64=0.01,
                 target_epsilon::Float64=5e-3)

    # Initialize output record structure 
    output_record = Vector{NamedTuple}(undef, steps)
    epsilon = epsilon_0

    for i_step in 1:steps
        println("Starting step $(i_step)")
        println("epsilon = $(epsilon)")

        if i_step == 1  # First ABC calculation
            result = basic_abc(model,
                               epsilon=epsilon,
                               max_iter=max_iter,
                               min_accepted=min_accepted,
                               pmc_mode=false)

            # Initial epsilon selection
            epsilon = select_epsilon(result.distances[:],
                                     epsilon,
                                     target_acc_rate=target_acc_rate)

            theta = result.theta_accepted
            tau_squared = 2 * cov(theta; dims=1)
            # Add stabilization
            tau_squared += 1e-6 * Matrix(I, size(tau_squared, 1), size(tau_squared, 2))
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
                                     total_iterations=steps)

            output_record[i_step] = (theta_accepted=theta,
                                     D_accepted=result.distances,
                                     n_accepted=result.n_accepted,
                                     n_total=result.n_total,
                                     epsilon=epsilon,
                                     weights=weights,
                                     tau_squared=tau_squared,
                                     eff_sample=effective_sample)
        end

        n_accept = output_record[i_step].n_accepted
        n_tot = output_record[i_step].n_total
        accept_rate = n_accept / n_tot
        println("Acceptance Rate = $(accept_rate)")
        current_theta = mean(output_record[i_step].theta_accepted, dims=1)
        println("Current theta = $(current_theta)")
        println("--------------------")

        if accept_rate < minAccRate
            println("epsilon = $(epsilon)")
            println("Acceptance Rate = $(accept_rate)")
            return output_record[1:i_step]
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
                    @inbounds covar[j, k] += (x[i, j] - xbar[j]) * (x[i, k] - xbar[k]) * w_norm[i]
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
"""
function select_epsilon(distances::Vector{Float64},
                        current_epsilon::Float64;
                        target_acc_rate::Float64=0.01,
                        current_acc_rate::Float64=0.0,
                        iteration::Integer=1,
                        total_iterations::Integer=100)
    # Filter out NaN and very large distances
    valid_distances = distances[.!isnan.(distances)]
    valid_distances = valid_distances[valid_distances.<10.0]

    if isempty(valid_distances)
        @warn "No valid distances for epsilon selection"
        return current_epsilon
    end

    # Compute quantiles
    q25 = sb.percentile(valid_distances, 25)
    q50 = sb.percentile(valid_distances, 50)
    q75 = sb.percentile(valid_distances, 75)

    # Get adaptive alpha value
    alpha = compute_adaptive_alpha(iteration,
                                   current_acc_rate,
                                   target_acc_rate,
                                   total_iterations=total_iterations)

    # Adaptive selection based on acceptance rate
    if current_acc_rate > 0.0
        if current_acc_rate > target_acc_rate * 1.1
            new_epsilon = max(q25, current_epsilon * (1.0 - alpha))
        elseif current_acc_rate < target_acc_rate * 0.9
            new_epsilon = min(q75, current_epsilon * (1.0 + alpha))
        else
            new_epsilon = current_epsilon
        end
    else
        new_epsilon = q50
    end

    return new_epsilon
end

"""
Compute adaptive alpha value based on iteration and convergence metrics
"""
function compute_adaptive_alpha(iteration::Integer,
                                current_acc_rate::Float64,
                                target_acc_rate::Float64;
                                alpha_max::Float64=0.9,
                                alpha_min::Float64=0.1,
                                total_iterations::Integer=100)
    # Base decay factor based on iteration progress
    progress = iteration / total_iterations
    base_alpha = alpha_max * (1 - progress) + alpha_min * progress

    # Adjust based on how far we are from target acceptance rate
    acc_rate_diff = abs(current_acc_rate - target_acc_rate) / target_acc_rate

    if acc_rate_diff > 2.0
        # Far from target: more aggressive adaptation
        alpha = min(alpha_max, base_alpha * 1.5)
    elseif acc_rate_diff < 0.2
        # Close to target: more conservative adaptation
        alpha = max(alpha_min, base_alpha * 0.5)
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
function find_MAP(theta_accepted::Matrix{Float64}, N::Integer)
    num_params = size(theta_accepted, 2)

    # Create grid of positions for each parameter
    positions = zeros(Float64, N, num_params)
    for i in 1:num_params
        param = @view theta_accepted[:,i]
        @inbounds positions[:,i] = rand(dist.Uniform(minimum(param), maximum(param)), N)
    end

    # Estimate density using KDE
    kernel = [kde(theta_accepted[:, i]) for i in 1:num_params]

    # Evaluate density at grid positions
    probs = [pdf(kernel[i], positions[:,i]) for i in 1:num_params]

    # Find position with maximum probability
    max_idx = [findmax(probs[i])[2] for i in 1:num_params]
    theta_map = [positions[max_idx[i], i] for i in 1:num_params]

    return theta_map
end

end # module
