# [Simulation Based Timescale Estimation](@id sim)

In simulation based methods, your data is assumed to come from a generative model and INT.jl performs Bayesian parameter estimation (via approximate Bayesian computation, ABC or automatic differentiation variational inference, ADVI) on that model. The goal is to match the autocorrelation function (ACF) or equivalently, power spectral density (PSD) of the generative model and data. The simplest generative model is an Ornstein-Uhlenbeck (OU) process with only one parameter to estimate. In case of oscillations, an oscillation is linearly added to the output of the Ornstein-Uhlenbeck process. If some of your data is missing, indicated by `NaN` or `missing`, the data points from the generative model are replaced by `NaN`s. We note that the variance of noise in the OU process is not fit to data as we scale the output of simulations to match the variance of data in order to reduce the burden or parameter fitting procedure. 

All methods assume that your data has one dimension for trials and one dimension for time points. From each trial, INT.jl calculates one summary statistic (ACF or PSD) and averages them across trials to get a less noisy estimate. The simulations from the generative model have the same data structure (same number of data points, trials and time resolution) as your data. The goal of the simulation based methods is minimizing the distance between the ACF or PSD of your model and data. Then the parameter corresponding to INT in your model is hopefully the real INT. 

## Implementation

There are four main functions in INT.jl to perform simulation based timescale estimation: [`one_timescale_model`](one_timescale.md), [`one_timescale_and_osc_model`](one_timescale_and_osc.md), [`one_timescale_with_missing_model`](one_timescale_with_missing.md), [`one_timescale_and_osc_with_missing_model`](one_timescale_and_osc_with_missing.md). For each model, one can choose between `:abc` or `:advi` as the inference method and `:acf` or `:psd` as the summary method. All models have the same syntax with differences in implementation. The detailed usage is documented in [`one_timescale_model`](one_timescale.md) - other model pages focus on specific differences. 

The following table summarizes the four models. 

| Model | Generative Model | Summary Method (`:acf` or `:psd`) | Supported Inference Methods (`:abc` or `:advi`) |
|-------|------------------|----------------|------------------|
| `one_timescale_model` | Ornstein-Uhlenbeck process | [`comp_ac_fft`](@ref) or [`comp_psd_adfriendly`](@ref) | ABC and ADVI |
| `one_timescale_and_osc_model` | Sinusoid added on Ornstein-Uhlenbeck process | [`comp_ac_fft`](@ref) or [`comp_psd_adfriendly`](@ref) | ABC and ADVI |
| `one_timescale_with_missing_model` | Ornstein-Uhlenbeck process with missing data replaced by NaNs | [`comp_ac_time_missing`](@ref) or [`comp_psd_lombscargle`](@ref) | ABC (for both ACF and PSD), ADVI (only ACF) |
| `one_timescale_and_osc_with_missing_model` | Sinusoid added on Ornstein-Uhlenbeck process with missing data replaced by NaNs | [`comp_ac_time_missing`](@ref) or [`comp_psd_lombscargle`](@ref) | ABC (for both ACF and PSD), ADVI (only ACF) |   

All models are fit with `fit` function and return [`ADVIResults`](@ref) or [`ABCResults`](@ref) type. See the [Fitting and Results](fit_result.md) section for details. 

## Fitting Methods - ABC

Approximate Bayesian Computation (ABC) is a method to approximate the posterior without solving the likelihood function. The algorithm has two steps: ABC ([`basic_abc`](@ref)) and population monte carlo (PMC, [`pmc_abc`](@ref)). In pseudocode, ABC is as follows:

```
summary = summary_statistic(empirical_data)
accepted_samples = []
WHILE length(accepted_samples) < min_accepted
    theta = sample_from_prior()
    model_data = simulate_data(model, theta)
    distance = compute_distance(summary, model_data)
    IF distance < epsilon
        push!(accepted_samples, theta)
    END IF
END WHILE
```

PMC uses ABC samples as the initial population and iteratively updates. For more details, refer to [Zeraati et al, 2021](https://www.nature.com/articles/s43588-022-00214-3). 

To change the parameters of the ABC algorithm, first use the function [`get_param_dict_abc`](@ref) to get the default parameters. Then modify the parameters and pass them to the function `fit`. For example, 

```julia
model = one_timescale_model(data, time, :abc)
param_dict = get_param_dict_abc()
param_dict["convergence_window"] = 10
result = fit(model, param_dict)
int_map = result.MAP[1] # Maximum a posteriori 
```
The parameters are detailed in [ABC Parameters](abc_parameters.md) section.

## Fitting Methods - ADVI

Automatic Differentiation Variational Inference (ADVI) approximates the posterior using variational methods. Instead of using MCMC directly, ADVI uses gradient descent to find the optimal parameters that minimize the Kullback-Leibler divergence between the variational posterior and the true posterior. INT.jl uses the [`Turing.jl`](https://turing.ml/stable/) package to perform ADVI. For more details, refer to [Turing documentation](https://turing.ml/v0.22/docs/for-developers/variational_inference) or [Kucukelbir et al, 2017](https://arxiv.org/abs/1603.00788). 

INT.jl states the probabilistic problem as the likelihood of each data point in the summary statistic of the data coming from a Gaussian distribution with mean generative model's summary statistic and some uncertainty around it. More clearly:

```math
\textrm{data summary statistic}_i \sim N(\textrm{model summary statistic}_i, \sigma)
```

Similar to ABC, in order to change the parameters of the ADVI algorithm, use the function [`get_param_dict_advi`](@ref) to get the default parameters, modify them, and pass them to the function `fit`. Example:

```julia
model = one_timescale_and_osc_model(data, time, :advi)
param_dict = get_param_dict_advi()
param_dict["n_iterations"] = 20
fit(model, param_dict)
```

See [ADVI Parameters](advi_parameters.md) section for details on parameters.

## Notes on Summary Statistics

Each model uses either ACF or PSD as the summary statistic. As can be seen from the table above, with no missing data, [`comp_ac_fft`](@ref) and [`comp_psd_adfriendly`](@ref) are used. [`comp_ac_fft`](@ref) calculates the ACF using the fast fourier transform (FFT). [`comp_psd_adfriendly`](@ref) is an autodifferentiable implementation of [`comp_psd`](@ref); both use Periodogram method with a Hamming window. In case of missing data, ACF is calculated in the time domain with the same techniques used in [statsmodels.tsa.stattools.acf](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html) with `missing=conservative` option. For PSD, Lomb-Scargle method (via [LombScargle.jl](https://juliaastro.org/LombScargle.jl/stable/)) with the function  [`comp_psd_lombscargle`](@ref) is used but currently it is not autodifferentiable. If you wish to use PSD with simulation-based inference in the case of missing data, you would need to use ABC. 
