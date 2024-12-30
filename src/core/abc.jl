# src/core/abc.jl
# TODO: Overall cleanup
module ABC

using Infiltrator
using ..Models
using Statistics
import Distributions as dist
import StatsBase as sb

export basic_abc, pmc_abc, effective_sample_size, weighted_covar

"""
Basic ABC rejection sampling algorithm
"""
function basic_abc(model::Models.AbstractTimescaleModel;
                   epsilon::Float64,
                   max_iter::Int,
                   pmc_mode::Bool=false,
                   weights=nothing,
                   theta_prev=nothing,
                   tau_squared=nothing)
    n_theta = length(model.prior)
    samples = zeros(n_theta, max_iter)
    isaccepted = zeros(max_iter)
    distances = zeros(max_iter)
    accepted_count = 0

    for trial_count in 1:max_iter

        # Draw from prior or proposal
        if pmc_mode && !isnothing(theta_prev)
            theta_star = theta_prev[:,
                                    sb.sample(collect(1:length(theta_prev)),
                                              sb.pweights(weights))]
            theta = rand(dist.MvNormal(theta_star, tau_squared))
            # Only sample positive values
            while sum(theta .< 0) > 0
                theta = rand(dist.MvNormal(theta_star, tau_squared))
            end
        else
            theta = Models.draw_theta(model)
        end

        # Generate data and compute distance
        d = Models.generate_data_and_reduce(model, theta)
        samples[:, trial_count] = theta
        distances[trial_count] = d

        if d <= epsilon
            accepted_count += 1
            isaccepted[trial_count] = 1
        end
    end # for
    theta_accepted = samples[:, isaccepted.==1]
    weights = ones(length(theta_accepted))
    tau_squared = zeros(length(theta_accepted), length(theta_accepted))
    eff_sample = length(theta_accepted)

    return (samples=samples,
            isaccepted=isaccepted,
            theta_accepted=theta_accepted,
            distances=distances,
            n_accepted=accepted_count,
            n_total=max_iter,
            epsilon=epsilon,
            weights=weights,
            tau_squared=tau_squared,
            eff_sample=eff_sample)
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

    weights_new = zeros(size(theta, 2))

    if length(size(theta)) == 1
        # Case for single parameter vector
        norm = zeros(length(theta))
        for (i, T) in enumerate(theta)
            for j in axes(theta_prev, 2)
                norm[j] = dist.pdf(dist.Normal(theta_prev[1, j], sqrt(tau_squared[1, 1])),
                                   T)
            end
            weights_new[i] = dist.pdf(prior[1], T) / sum(weights .* norm)
        end

        return weights_new ./ sum(weights_new)

    else
        # Case for multiple parameter vectors
        norm = zeros(size(theta_prev, 2))
        for i in axes(theta, 2)
            prior_prob = zeros(size(theta, 1))
            for j in axes(theta, 1)
                prior_prob[j] = dist.pdf(prior[j], theta[j, i])
            end
            # Assumes independent priors
            p = prod(prior_prob)

            for j in axes(theta_prev, 2)
                norm[j] = dist.pdf(dist.MvNormal(theta_prev[:, j], tau_squared),
                                   theta[:, i])
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
function weighted_covar(x::Union{Vector{Float64}, Matrix{Float64}}, w::Vector{Float64})
    sumw = sum(w)
    @assert isapprox(sumw, 1.0)

    if ndims(x) == 1
        @assert length(x) == length(w)
    else
        @assert size(x, 2) == length(w)
    end

    sum2 = sum(w .^ 2)

    if ndims(x) == 1
        xbar = sum(w .* x)
        var = sum(w .* (x .- xbar) .^ 2)
        return var * sumw / (sumw * sumw - sum2)
    else
        xbar = [sum(w .* x[i, :]) for i in axes(x, 1)]
        covar = zeros(size(x, 1), size(x, 1))
        for k in axes(x, 1)
            for j in axes(x, 1)
                for i in axes(x, 2)
                    covar[j, k] += (x[j, i] - xbar[j]) * (x[k, i] - xbar[k]) * w[i]
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

# TODO: Implement parallelism
"""
pmc_abc(model, data, inter_save_direc, inter_filename; 
        epsilon_0=1.0, min_samples=10, steps=10, resume=nothing, 
        parallel=false, n_procs="all", sample_only=false, 
        minError=0.0001, minAccRate=0.0001)

Perform a sequence of ABC posterior approximations using the sequential population Monte Carlo algorithm.

# Arguments
- `model`: Model object that is a subclass of AbstractTimescaleModel
- `data`: The "observed" data set for inference
- `inter_save_direc`: Directory to save intermediate results
- `inter_filename`: Filename for intermediate results
- `epsilon_0=1.0`: Initial tolerance to accept parameter draws
- `min_samples=10`: Minimum number of posterior samples
- `steps=10`: Number of PMC steps to attempt
- `resume=nothing`: Record array of previous PMC sequence to continue from
- `parallel=false`: Whether to run in parallel mode
- `n_procs="all"`: Number of processes for parallel mode, "all" uses all cores
- `sample_only=false`: Whether to only sample without computing weights
- `minError=0.0001`: Minimum error threshold
- `minAccRate=0.0001`: Minimum acceptance rate threshold

# Returns
A record array containing ABC output for each step with fields:
- `theta accepted`: Array of posterior samples
- `D accepted`: Array of accepted distances  
- `n accepted`: Number of accepted samples
- `n total`: Total number of samples attempted
- `epsilon`: Distance tolerance used
- `weights`: Importance sampling weights (array of 1s if not in PMC mode)
- `tau_squared`: Gaussian kernel variances (array of 0s if not in PMC mode)
- `eff sample`: Effective sample size (array of 1s if not in PMC mode)
"""
function pmc_abc(model::Models.AbstractTimescaleModel;
                 epsilon_0::Float64=1.0, min_samples::Int=10, steps::Int=10,
                 sample_only::Bool=false, max_iter::Int=10000, minAccRate::Float64=0.0001)

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
                               pmc_mode=false)
            theta = result.theta_accepted
            tau_squared = 2 * cov(theta; dims=2)
            weights = fill(1.0 / size(theta, 2), size(theta, 2))
            epsilon = sb.percentile(result.distances, 75)
            eff_sample = effective_sample_size(weights)

            output_record[i_step] = (theta_accepted=theta,
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
                               pmc_mode=true,
                               weights=weights_prev,
                               theta_prev=theta_prev,
                               tau_squared=tau_squared)

            theta = result.theta_accepted
            epsilon = sb.percentile(result.distances, 75)
            effective_sample = effective_sample_size(weights_prev)

            if sample_only
                weights = Float64[]
                tau_squared = zeros(0, 0)
            else
                weights = calc_weights(theta_prev, theta, tau_squared,
                                       weights_prev, model.prior)
                tau_squared = 2 * weighted_covar(theta, weights)
            end

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
        println("--------------------")

        if accept_rate < minAccRate
            println("epsilon = $(epsilon)")
            println("Acceptance Rate = $(accept_rate)")
            return output_record[1:i_step]
        end
    end

    return output_record
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
# TODO: Figure out KDE
# function find_MAP(theta_accepted::Matrix{Float64}, N::Int)
#     num_params = size(theta_accepted, 1)
    
#     # Create grid of positions for each parameter
#     positions = zeros(num_params, N)
#     for i in 1:num_params
#         param = @view theta_accepted[i,:]
#         positions[i,:] = rand(Uniform(minimum(param), maximum(param)), N)
#     end
    
#     # Estimate density using KDE
#     kernel = kde(theta_accepted)
    
#     # Evaluate density at grid positions
#     probs = pdf(kernel, positions)
    
#     # Find position with maximum probability
#     _, max_idx = findmax(probs)
#     theta_map = positions[:,max_idx]
    
#     return theta_map
# end

end # module
