module TuringBackend

using Turing
using Statistics
using IntrinsicTimescales
using ..Models

export fit_vi, ADVIResults, get_param_dict_advi

"""
    ADVIResults{T<:Real}

Container for ADVI (Automatic Differentiation Variational Inference) results.

# Fields
- `samples::AbstractArray{T}`: Matrix of posterior samples
- `MAP::AbstractVector{T}`: Maximum a posteriori estimates
- `variational_posterior::Any`: Turing variational posterior object containing full inference results
"""
struct ADVIResults{T<:Real}
    samples::AbstractArray{T}
    MAP::AbstractVector{T}
    variational_posterior::Any
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
            theta[i] ~ truncated(model.prior[i], 0.0, Inf)
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
- `ADVIResults`: Container with inference results including:
  - `samples_matrix`: Matrix of posterior samples
  - `MAP`: Maximum a posteriori parameter estimates
  - `variational_posterior`: Turing variational posterior object containing full inference results

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

    return ADVIResults(samples_matrix, MAP, chain)
end



"""
    get_param_dict_advi()

Get default parameter dictionary for ADVI (Automatic Differentiation Variational Inference) algorithm.

# Returns
Dictionary containing default values for ADVI parameters including:
- `n_samples`: Number of posterior samples to draw (default: 4000)
- `n_iterations`: Number of ADVI iterations (default: 50)
- `n_elbo_samples`: Number of samples for ELBO estimation (default: 20)
- `autodiff`: Automatic differentiation backend (default: AutoForwardDiff())
"""
function get_param_dict_advi()
    return Dict(
        :n_samples => 4000,
        :n_iterations => 50,
        :n_elbo_samples => 20,
        :autodiff => AutoForwardDiff()
    )
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
