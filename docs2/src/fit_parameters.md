# Model Fitting and Parameters

All models are fit with `fit` function and return [`ADVIResults`](@ref) or [`ABCResults`](@ref) type. The `fit` function has the following signature:

```julia
results = fit(model, param_dict=nothing)
```

The function determines the inference method based on model attributes which the user provides when initiating the model. When `param_dict` is not provided, the function uses the default parameters for the inference method, which can be seen with [`get_param_dict_advi`](@ref) and [`get_param_dict_abc`](@ref) functions. 

## Parameters for Approximate Bayesian Computation (ABC)

The parameters of ABC can be accessed and modified through the `get_param_dict_abc()` function. 

#### General ABC Parameters
- `epsilon_0::Float64 = 1.0`: Initial acceptance threshold. If the distance between the observed data and the simulated data is less than `epsilon_0`, the sample is accepted in the initial step of ABC. Subsequent steps change the epsilon value to adapt better. 
- `max_iter::Int = 10000`: Maximum number of iterations per basic ABC step
- `min_accepted::Int = 100`: The number of accepted samples for basic ABC
- `steps::Int = 30`: Number of PMC steps to perform
- `sample_only::Bool = false`: If true, only perform sampling without adaptation between basic ABC runs

#### Epsilon Selection Parameters

Different from the Zeraati et al. (2022) method, we adaptively change the epsilon value between basic ABC steps. The epsilon selection procedure adaptively adjusts the acceptance threshold based on the current acceptance rate and distance distribution. The procedure works as follows:

1. First, invalid distances (NaN values and distances above `distance_max`) are filtered out.

2. Three quantiles are computed from the valid distances:
   - Lower quantile (`quantile_lower`) for conservative threshold
   - Initial quantile (`quantile_init`) for first iteration
   - Upper quantile (`quantile_upper`) for relaxed threshold

3. An adaptive alpha value is computed based on:
   - Progress through iterations (iteration/total_iterations) to decay from alpha_max to alpha_min
   - Difference between current and target acceptance rates:
     - If difference > acc_rate_far: Alpha increases to min(alpha_max, base_alpha * alpha_far_mult)
     - If difference < acc_rate_close: Alpha decreases to max(alpha_min, base_alpha * alpha_close_mult) 
     - Otherwise: Uses base alpha from iteration progress
   
4. The new epsilon is then selected:
   - For first iteration: Uses the initial quantile
   - For subsequent iterations:
     - If acceptance rate is too high: Epsilon is set to the maximum of the lower quantile and epsilon * (1-alpha)
     - If acceptance rate is too low: Epsilon is set to the minimum of the upper quantile and epsilon * (1+alpha) 
     - If acceptance rate is within buffer of target: Epsilon stays same

This adaptive procedure helps balance exploration and exploitation during the ABC sampling process by sampling wider for initial steps and narrowing down as the algorithm converges. 

Parameters controlling epsilon selection:

- `target_acc_rate::Float64 = 0.01`: Targeted acceptance rate for epsilon adaptation
- `distance_max::Float64 = 10.0`: Maximum distance to consider valid
- `quantile_lower::Float64 = 25.0`: Lower quantile for epsilon adjustment
- `quantile_upper::Float64 = 75.0`: Upper quantile for epsilon adjustment
- `quantile_init::Float64 = 50.0`: Initial quantile when no acceptance rate
- `acc_rate_buffer::Float64 = 0.1`: Buffer around target acceptance rate

#### Adaptive Alpha Parameters
- `alpha_max::Float64 = 0.9`: Maximum adaptation rate
- `alpha_min::Float64 = 0.1`: Minimum adaptation rate
- `acc_rate_far::Float64 = 2.0`: Threshold for "far from target" adjustment
- `acc_rate_close::Float64 = 0.2`: Threshold for "close to target" adjustment
- `alpha_far_mult::Float64 = 1.5`: Multiplier for alpha when far from target
- `alpha_close_mult::Float64 = 0.5`: Multiplier for alpha when close to target

#### Early Stopping Parameters
- `convergence_window::Int = 3`: Number of iterations to check for convergence
- `theta_rtol::Float64 = 1e-2`: Relative tolerance for parameter convergence
- `theta_atol::Float64 = 1e-3`: Absolute tolerance for parameter convergence
- `target_epsilon::Float64 = 5e-3`: Stop the PMC if the distance between the observed data and the simulated data is less than `target_epsilon`.
- `minAccRate::Float64 = 0.01`: If acceptance rate of basic ABC steps is below `minAccRate`, the algorithm stops.

#### Numerical Stability Parameters
- `jitter::Float64 = 1e-6`: Small value added to covariance diagonal for numerical stability
- `cov_scale::Float64 = 2.0`: Scaling factor for covariance matrix to calculate `tau_squared` which is used to calculate the weights of posterior samples for calculating the new prior. 

#### Display Parameters
- `show_progress::Bool = true`: Whether to show progress bar
- `verbose::Bool = true`: Whether to print detailed information

#### MAP Estimation Parameters
The `find_MAP` function estimates the maximum a posteriori (MAP) parameters by performing a grid search over the parameter space. It takes the accepted parameters from the final ABC step and creates a grid of N random positions within the parameter bounds. For each parameter dimension, it estimates the probability density using kernel density estimation (KDE) and evaluates the density at the grid positions. The MAP estimate is then determined by finding the position with maximum probability density for each parameter. This provides a point estimate of the most probable parameter values given the posterior samples.

- `N::Int = 10000`: Number of samples for maximum a posteriori (MAP) estimation grid search

To modify these parameters, create a dictionary with your desired values and pass it to the `fit` function:

```julia
model = one_timescale_model(data, time, :abc)
param_dict = get_param_dict_abc()
param_dict[:convergence_window] = 10
param_dict[:max_iter] = 20000
results = fit(model, param_dict)
```

## Parameters for Automatic Differentiation Variational Inference (ADVI)

ADVI is performed via [`Turing.jl`](https://turing.ml/v0.22/docs/for-developers/variational_inference) package. See the [variational inference tutorial](https://turing.ml/dev/tutorials/09-variational-inference/) to learn more about Turing's ADVI implementation. The parameters can be accessed and modified through the `get_param_dict_advi()` function. 

- `n_samples::Int = 4000`: Number of posterior samples to draw after fitting
- `n_iterations::Int = 50`: Number of ADVI optimization iterations. Increase this number if your model is not fitting well.
- `n_elbo_samples::Int = 20`: Number of samples used to estimate the ELBO (Evidence Lower BOund) during optimization. Increase this number if your model is not fitting well.
- `autodiff = AutoForwardDiff()`: The automatic differentiation backend to use for computing gradients. Currently, only `AutoForwardDiff()` is supported.

To modify these parameters, create a dictionary with your desired values and pass it to the `fit` function:

```julia
model = one_timescale_model(data, time, :advi)
param_dict = get_param_dict_advi()
param_dict[:n_samples] = 8000
param_dict[:n_elbo_samples] = 60
results = fit(model, param_dict)
```

