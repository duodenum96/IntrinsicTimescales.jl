# [One Timescale Model (`one_timescale_model`)](@id one_timescale)

The generative model:

```math
\frac{dx}{dt} = -\frac{x}{\tau} + \xi(t)
```

with timescale ``\tau``. ``\xi(t)`` is white noise with unit variance. 

Can be used with either ACF or PSD as the summary method. Below, in each section, we describe the arguments for each summary method. 

## ACF Summary

The full function signature for `summary_method=:acf` is:

```julia
function one_timescale_model(data, time, fit_method; summary_method=:acf,
                             prior=nothing, n_lags=nothing,
                             distance_method=nothing,
                             dims=ndims(data), distance_combined=false,
                             weights=[0.5, 0.5])
```

Example usage:

```julia
# Simulate some data:
using IntrinsicTimescales
timescale = 0.3 # true timescales
variance = 1.0 # variance of data
duration = 10.0 # duration of data
n_trials = 10 # How many trials
fs = 500.0 # Sampling rate
data = generate_ou_process(timescale, variance, 1/fs, duration, n_trials) # Data in the form of (trials x time)

# Prepare the vector of time points:
time = (1/fs):(1/fs):duration

# Fit the model:
model = one_timescale_model(data, time, :abc, summary_method=:acf)
results = int_fit(model)
int = results.MAP[1] # maximum a posteriori estimate
```

#### Mandatory arguments: 

* `data`: Your time-series data as a vector or 2-dimensional array. 

If it is n-dimensional, by default, the dimension of time is assumed to be the last dimension. If this is not the case, you can set it via `dims` argument, similar to [`acw`](acw.md) function. The other dimension should correspond to trials. IntrinsicTimescales.jl calculates one ACF from each trial and averages them to get a less noisy ACF estimate. If the user wants to calculate one INT per trial, they can run a for-loop over trials. 

* `time`: Time points corresponding to the data. 

* `fit_method`: `:abc` or `:advi`. Method to use for parameter estimation. 

#### Optional arguments: 

* `summary_method`: `:acf`. Method to use for summary statistics. 

If `:acf`, calculates the autocorrelation function using `comp_ac_fft` internally. If `:psd`, calculates the power spectral density using `comp_psd`. 

* `prior`: Prior distribution for the parameters. `"informed_prior"` or a Distribution object from [Distributions.jl](https://juliastats.org/Distributions.jl/stable/). 

If the user does not specify a prior, or specifies `"informed_prior"`, IntrinsicTimescales.jl uses a normal distribution with mean determined by fitting an exponential decay to the autocorrelation function using `fit_expdecay` and standard deviation of 20. Currently we recommend explicitly specifying a prior distribution to improve the accuracy of the inference. 

An example for custom prior distribution:

```julia
using Distributions
prior = Normal(0.5, 0.5)
model = one_timescale_model(data, time, :abc, summary_method=:acf, prior=prior)
results = int_fit(model)
int = results.MAP[1]
```

* `n_lags`: Number of lags to use for the ACF calculation. 

By default, this is set to `1.1*acw0`. The reason for cutting the autocorrelation function is due to increased measurement noise for autocorrelation function for later lags. Intuitively, when we perform correlation of a time-series with a shifted version of itself, we have less and less number of time points to calculate the correlation. This is why if you plot the autocorrelation function, the portion of it after ACW-0 looks more noisy. For more details, see [Practice 1]. 

* `distance_method`: `:linear` or `:logarithmic`. 

Method to use for distance calculation. `:linear` is RMSE between the ACF from data and the model ACF whereas `:logarithmic` is RMSE after log-transforming the ACF. The default is `:linear`. 

* `distance_combined`: `true` or `false`. 

If `true`, the distance is a weighted sum of RMSE between ACFs and RMSE between exponential decay fits to ACFs. Defaults to `false`.

* `weights`: A vector of two numbers. 

The first number is the weight for RMSE between ACFs and the second number is the weight for RMSE between exponential decay fits to ACFs. The default is `[0.5, 0.5]`. Used only if `distance_combined` is `true`. 

## PSD Summary

The full function signature for `summary_method=:psd` is:

```julia
function one_timescale_model(data, time, fit_method; summary_method=:psd,
                             prior=nothing, 
                             distance_method=nothing, freqlims=nothing,
                             dims=ndims(data), distance_combined=false,
                             weights=[0.5, 0.5])
```

Example usage:

```julia
model = one_timescale_model(data, time, :abc, summary_method=:psd)
results = int_fit(model)
int = results.MAP[1]
```


There are two arguments that are different from the ACF summary:

* `summary_method`: `:psd`. Method to use for summary statistics. 

Calculates the power spectral density using `comp_psd`. 

* `freqlims`: A tuple of two numbers. 

The first number is the lower frequency limit and the second number is the upper frequency limit used to index the power spectral density. The default is the output from [`fftfreq`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.fftfreq) function of [AbstractFFTs.jl](https://juliamath.github.io/AbstractFFTs.jl/stable/) library. 

#### Implementation differences from ACF Summary

* `prior`: `"informed_prior"` fits a lorentzian to the PSD and calculates the INT from the knee frequency using `tau_from_knee` and `find_knee_frequency`. 

* `distance_method`: `:linear` is RMSE between the PSD from data and the model PSD whereas `:logarithmic` is RMSE after log-transforming the PSD. The default is `:linear`. 

* `distance_combined`: Weighted sum between RMSE between PSDs and RMSE between INT estimates from knee frequency obtained from lorentzian fits to PSD. 

* `weights`: A vector of two numbers. 

The first number is the weight for RMSE between PSDs and the second number is the weight for RMSE between INT estimates. 

## Returns

* `model`: A `OneTimescaleModel` object. Can be used as an input to `fit` function to estimate INT and `posterior_predictive` function to plot posterior predictive check. 
