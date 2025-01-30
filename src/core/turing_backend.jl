module TuringBackend

using Turing
using Statistics
using BayesianINT
using ..Models

export fit_vi, TuringResult

"""
    TuringResult{T<:Real}

Container for ADVI (Automatic Differentiation Variational Inference) results.

# Fields
- `samples::AbstractArray{T}`: Matrix of posterior samples
- `MAP::AbstractVector{T}`: Maximum a posteriori estimates
- `variances::AbstractVector{T}`: Posterior variances for each parameter
- `chain`: Turing chain object containing full inference results
"""
struct TuringResult{T<:Real}
    samples::AbstractArray{T}
    MAP::AbstractVector{T}
    variances::AbstractVector{T}
    chain::Any
end

"""
    create_turing_model(model, data_sum_stats; σ_prior=Exponential(1))

Create a Turing probabilistic model for variational inference.

# Arguments
- `model`: Model instance containing prior distributions and data generation methods
- `data_sum_stats`: Summary statistics of the observed data
- `σ_prior=Exponential(1)`: Prior distribution for the uncertainty parameter σ

# Returns
- Turing model object ready for inference

# Notes
The created model includes:
- Parameter sampling from truncated priors (positive values only)
- Data generation using the model's forward simulation
- Likelihood computation using Normal distribution
"""
function create_turing_model(model, data_sum_stats; σ_prior=Exponential(1))
    Turing.@model function fit_summary_stats(model, data)
        # Get priors from model object
        theta = Vector{Real}(undef, length(model.prior))
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
    fit_vi(model; n_samples=4000, n_iterations=10, n_elbo_samples=20, 
           optimizer=AutoForwardDiff())

Perform variational inference using ADVI (Automatic Differentiation Variational Inference).

# Arguments
- `model`: Model instance to perform inference on
- `n_samples::Int=4000`: Number of posterior samples to draw
- `n_iterations::Int=10`: Number of ADVI iterations
- `n_elbo_samples::Int=20`: Number of samples for ELBO estimation
- `optimizer=AutoForwardDiff()`: Optimization algorithm for ADVI

# Returns
- `TuringResult`: Container with inference results including:
  - Posterior samples
  - MAP estimates
  - Parameter variances
  - Full Turing chain

# Notes
Uses Turing.jl's ADVI implementation for fast approximate Bayesian inference.
The model is automatically constructed with appropriate priors and likelihood.
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