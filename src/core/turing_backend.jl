module TuringBackend

using Turing
using Statistics
using BayesianINT
using ..Models

export fit_vi, TuringResult

struct TuringResult{T<:Real}
    samples::AbstractArray{T}
    MAP::AbstractVector{T}
    variances::AbstractVector{T}
    chain::Any
end


"""
    create_turing_model(model, data_sum_stats; σ_prior=Exponential(1))

Creates a Turing model for the given model object and summary statistics.
"""
function create_turing_model(model, data_sum_stats; σ_prior=Exponential(1))
    Turing.@model function fit_summary_stats(model, data)
        # Get priors from model object
        theta = Vector(undef, length(model.prior))
        for i in eachindex(model.prior)
            theta[i] ~ Truncated(model.prior[i], 0.0, Inf)
        end

        σ ~ σ_prior
        # Generate data and compute summary statistics
        sim_data = Models.generate_data(model, theta)
        predicted_stats = Models.summary_stats(model, sim_data)
        
        # Likelihood
        for i in eachindex(data)
            data[i] ~ Normal(predicted_stats[i], σ^2)
        end
    end
    
    return fit_summary_stats(model, data_sum_stats)
end

"""
    TuringResult{T<:Real}

Type-parameterized struct to hold ADVI results.
"""

"""
    fit_vi(model; n_samples=4000, n_iterations=10, n_elbo_samples=20, 
           optimizer=AutoForwardDiff())

Fits a model using Variational Inference through Turing.
"""
function fit_vi(model; n_samples=4000, n_iterations=10, n_elbo_samples=20, 
                optimizer=AutoForwardDiff())
    # Create and fit Turing model
    turing_model = create_turing_model(model, model.data_sum_stats)
    advi = ADVI(n_iterations, n_elbo_samples, optimizer)
    chain = vi(turing_model, advi)
    
    # Draw samples and compute statistics
    samples = rand(chain, n_samples)
    samples_matrix = Matrix(samples)
    
    # Compute MAP and variances
    MAP = find_MAP(samples_matrix')
    variances = vec(var(samples_matrix, dims=1))
    
    return TuringResult(samples_matrix, MAP, variances, chain)
end

end # module

# TODO: Make this type stable
# import Random
# @code_warntype turing_model.f(
#     turing_model,
#     Turing.VarInfo(turing_model),
#     Turing.SamplingContext(
#         Random.default_rng(), Turing.SampleFromPrior(), Turing.DefaultContext()
#     ),
#     turing_model.args...,
# )