# Results 

The `fit` function returns a [`ADVIResults`](@ref) or [`ABCResults`](@ref) object. 

## ABCResults

The ABC algorithm returns an `ABCResults` type containing the full history and final results of the inference process. The type has the following fields:

- `MAP::Vector`: Maximum a posteriori estimates of the parameters.
- `theta_history::Vector{Matrix}`: History of parameter values across all PMC iterations. Each matrix contains the accepted parameters for that iteration with columns being parameters and rows being samples.
- `epsilon_history::Vector`: History of acceptance thresholds (epsilon values) used in each iteration.
- `acc_rate_history::Vector`: History of acceptance rates achieved in each iteration.
- `weights_history::Vector{Vector}`: History of importance weights for accepted samples in each iteration.
- `final_theta::Matrix`: Final accepted parameter values from the last iteration.
- `final_weights::Vector`: Final importance weights from the last iteration.

You can access these fields directly from the results type:

```julia
results = fit(model, param_dict)

# Maximum a posteriori estimates
map_estimates = results.MAP

# Access final parameter values
final_params = results.final_theta

# Get acceptance rates across iterations
acc_rates = results.acc_rate_history

# Get MAP estimates
map_estimates = results.MAP
```

## ADVIResults

The ADVI algorithm returns an `ADVIResults` type containing the inference results. The type has the following fields:

- `samples::AbstractArray`: Matrix of posterior samples drawn after fitting. Each row represents a sample and each column represents a parameter.
- `MAP::AbstractVector`: Maximum a posteriori estimates of the parameters.
- `variational_posterior`: The fitted variational posterior distribution containing the full inference results. In `Turing.jl`, this is obtained via [`q = vi(model, vi_alg)`](https://turing.ml/dev/tutorials/09-variational-inference/).   

You can access these fields directly from the results object:

```julia
results = fit(model, param_dict)
posterior_samples = results.samples # Access posterior samples
map_estimates = results.MAP # Get MAP estimates
posterior = results.variational_posterior # Full variational posterior
```
