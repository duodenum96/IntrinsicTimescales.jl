# [Simulation Based Timescale Estimation](@id sim)

There are four main functions in INT.jl to perform simulation based timescale estimation: [`one_timescale_model`](one_timescale.md), [`one_timescale_and_osc_model`](one_timescale_and_osc.md), [`one_timescale_with_missing_model`](one_timescale_with_missing.md), [`one_timescale_and_osc_with_missing_model`](one_timescale_and_osc_with_missing.md). For each model, one can choose between `:abc` or `:advi` as the inference method and `:acf` or `:psd` as the summary method. All models have similar syntax with differences in implementation. For more details, refer to the [Practice] section. The following table summarizes the four models. 

| Model | Generative Model | Summary Method (`:acf` or `:psd`) | Supported Inference Methods (`:abc` or `:advi`) |
|-------|------------------|----------------|------------------|
| `one_timescale_model` | Ornstein-Uhlenbeck process | `comp_ac_fft` or `comp_psd` | ABC and ADVI |
| `one_timescale_and_osc_model` | Sinusoid added on Ornstein-Uhlenbeck process | `comp_ac_fft` or `comp_psd` | ABC and ADVI |
| `one_timescale_with_missing_model` | Ornstein-Uhlenbeck process with missing data replaced by NaNs | `comp_ac_time_missing` or `comp_psd_lombscargle` | ABC (for both ACF and PSD), ADVI (only ACF) |
| `one_timescale_and_osc_with_missing_model` | Sinusoid added on Ornstein-Uhlenbeck process with missing data replaced by NaNs | `comp_ac_time_missing` or `comp_psd_lombscargle` | ABC (for both ACF and PSD), ADVI (only ACF) |

## Fitting Methods - ABC

Approximate Bayesian Computation (ABC) is a method to approximate the posterior without solving the likelihood function. The algorithm has two steps: ABC ([`basic_abc`](@ref)) and population monte carlo (PMC, [`pmc_abc`](@ref)). In pseudocode, ABC is as follows:

```
summary = summary_statistic(empirical_data)
accepted_samples = []
WHILE length(accepted_samples) < min_accepted
    theta = sample_prior()
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
param_dict = get_param_dict_abc()
param_dict["convergence_window"] = 10
fit(model, data, param_dict)
```
The parameters are detailed in [ABC Parameters](abc_parameters.md) section.

## Fitting Methods - ADVI

Automatic Differentiation Variational Inference (ADVI) approximates the posterior using variational methods. Instead of using MCMC directly, ADVI uses gradient descent to find the optimal parameters that minimize the Kullback-Leibler divergence between the variational posterior and the true posterior. For more details, refer to [Kucukelbir et al, 2017](https://arxiv.org/abs/1603.00788). 

INT.jl uses the [`Turing.jl`](https://turing.ml/stable/) package to perform ADVI. 

Similar to ABC, in order to change the parameters of the ADVI algorithm, use the function [`get_param_dict_advi`](@ref) to get the default parameters, modify them, and pass them to the function `fit`. Example:

```julia
param_dict = get_param_dict_advi()
param_dict["n_iterations"] = 20
fit(model, data, param_dict)
```
