# [Model-Free Timescale Estimation](@id acw)

Performed via the function `acw` in IntrinsicTimescales.jl. The `acw` function calculates ACF or PSD depending on the acwtypes you specify. If there is no missing data (indicated by `NaN` or `missing`), `acw` calculates ACF as the inverse fourier transform of the power spectrum, using `comp_ac_fft` internally. Otherwise it calculates ACF as correlations between a time-series and its lag-shifted variants, using `comp_ac_time_missing`. For PSD, it uses periodogram method (`comp_psd`) in the case of no missing data and Lomb-Scargle method (`comp_psd_lombscargle`) in the case of missing data. 

```julia
acwresults = acw(data, fs; acwtypes=[:acw0, :acw50, :acweuler, :auc, :tau, :knee], 
                n_lags=nothing, freqlims=nothing, dims=ndims(data), 
                return_acf=true, return_psd=true, 
                average_over_trials=false, trial_dims=setdiff([1, 2], dims)[1], 
                skip_zero_lag=false,
                max_peaks=1, oscillation_peak::Bool=true,
                allow_variable_exponent::Bool=false,
                constrained::Bool=false,
                parallel::Bool=false,
                solver=LevenbergMarquardt,
                solver_kwargs)
```

Simple usage:

```julia
data = randn(10, 300) # 10 trials, 300 time points
fs = 1.0 # sampling rate
results = acw(data, fs)
acw_results = results.acw_results
acw_0 = acw_results[1]
acw_50 = acw_result[2]
# And so on...
```
## Arguments

#### Mandatory arguments: 

* `data`: Your time-series data as a vector or n-dimensional array. 

If it is n-dimensional, by default, the dimension of time is assumed to be the last dimension. For example, if you have a 2D array where rows are subjects and columns are time points, `acw` function will correctly assume that the last (2nd) dimension is time. If the dimension of time is any other dimension than the last, you can set it via `dims` argument. For example: 

```julia
data = randn(100, 200, 5) # 100 trials, 200 time points, 5 channels
acw(data, fs; dims=2)
```

* `fs`: Sampling rate of your data. A floating point number. 

#### Optional arguments:

* `acwtypes`: A symbol or a vector of symbols denoting which ACW types to calculate. 
In Julia, a symbol is written as `:symbol`. Example:

```julia
acw(data, fs; acwtypes=:acw0)
acw(data, fs; acwtypes=[:acw0, :acw50])
```

Supported ACW types:

`:acw0`: The lag where autocorrelation function crosses 0.

`:acw50`: The lag where autocorrelation function crosses 0.5.

`:acweuler`: The lag where autocorrelation function crosses ``1/e``. Corresponds to the inverse decay rate of an exponential decay function. 

`:tau`: Fit an exponential decay function ``e^{\frac{t}{\tau}}`` to the autocorrelation function and extract ``\tau``, which is the inverse decay rate. The parameter `skip_zero_lag` is used to specify whether to skip the zero lag for fitting an exponential decay function. When zero-lag is skipped, the function fits an exponential decay of the form ``A (exp(-lags / tau) + B)`` where A is the amplitude and B is the offset. See below for details and references. 

`:auc`: Calculate the area under the autocorrelation function from lag 0 to the lag where autocorrelation function crosses 0. 

`:knee`: Fit a lorentzian function ``\frac{A}{1 + (f/a)^2}`` to the power spectrum using an iterative FOOOF-style approach. By Wiener-Khinchine theorem, this is the power spectrum of a time-series with an autocorrelation function of exponential decay form. The parameter ``a`` corresponds to the knee frequency. ``\tau`` and ``a`` has the relationship ``\tau = \frac{1}{2 \pi a}``. The `:knee` method uses this relationship to estimate ``\tau`` from the knee frequency. In practice, first, an initial lorentzian fit is performed. Then, any oscillatory peaks are identified and fitted with gaussian functions. These gaussians are subtracted from the original power spectrum to ensure the remaining PSD is closer to a Lorentzian, and a final Lorentzian is fit to this "cleaned" spectrum. `freqlims` is used to specify the frequency limits to fit the lorentzian function. You can set the maximum number of oscillatory peaks to fit with the `max_peaks` argument. The argument `oscillation_peak` is used to specify whether to fit the oscillatory peaks or not. If set to `false`, just fit a Lorentzian and return the timescale estimated from the knee frequency. The parameter `allow_variable_exponent` is used to specify whether to allow a variable exponent in the lorentzian fit (i.e. ``b`` in ``\frac{A}{1 + (f/a)^b}``). This might be useful for cases where the power spectrum does not conform to a simple Lorentzian. The parameter `constrained` is used to specify whether to use constrained optimization when fitting a lorentzian to the PSD for knee frequency estimation (see below). 

* `n_lags`: An integer. Only used when `:tau` is in `acwtypes`. The number of lags to be used for fitting an exponential decay function. 

By default, this is set to `1.1*acw0`. The reason for cutting the autocorrelation function is due to increased measurement noise for autocorrelation function for later lags. Intuitively, when we perform correlation of a time-series with a shifted version of itself, we have less and less number of time points to calculate the correlation. This is why if you plot the autocorrelation function, the portion of it after ACW-0 looks more noisy. For more details, see [Practice 1]. 

* `freqlims`: Only used when `:knee` is in `acwtypes`. The frequency limits to fit the lorentzian function. 

By default, the lowest and highest frequencies that can be estimated from your data. A tuple of two floating point numbers, for example, `(freq_low, freq_high)` or `(1.0, 50.0)`. 

* `dims`: The dimension of time in your data. See above, the `data` argument for the explanation. An integer. 

* `return_acf`: Whether or not to return the autocorrelation function (ACF) in the results object. A boolean. 

* `return_psd`: Whether or not to return the power spectrum (psd) in the results object. A boolean. 

* `average_over_trials`: Whether or not to average the ACF or PSD across trials, as in [Honey et al., 2012]. 

Assuming that your data is stationary, averaging over trials can greatly reduce noise in your ACF/PSD estimations. By default, the dimension of trials is assumed to be the first dimension of your data. For example, if your data is two dimensional with rows as trials and columns as time points, the function will correctly infer the dimension of trials. If this is not the case, set the dimension of trials with the argument `trial_dims`. Below is an example of a three dimensional data with time as second and trials as third argument:

```julia
data = randn(10, 1000, 20) # 10 subjects, 1000 time points, 20 trials
result = acw(data, fs; dims=2, average_over_trials=true, trial_dims=3)
```

* `trial_dims`: Dimension of trials to average over. See above (`average_over_trials`) for explanation. An integer.

* `skip_zero_lag`: Whether or not to skip the zero lag for fitting an exponential decay function. Default is `false`. If true, the function will fit an exponential decay of the form ``A (exp(-lags / tau) + B)`` where A is the amplitude and B is the offset. This can be useful for cases with very low sampling rate (e.g. fMRI). The technique is used in [Ito et al., 2020](https://www.sciencedirect.com/science/article/pii/S1053811920306273) and [Murray et al., 2014](https://www.nature.com/articles/nn.3862). 

* `max_peaks`: Maximum number of oscillatory peaks to fit when cleaning the PSD for knee frequency estimation. Default is 1. 

* `oscillation_peak`: Whether or not to fit the oscillatory peaks when cleaning the PSD for knee frequency estimation. Default is `true`.

* `allow_variable_exponent`: Whether or not to allow variable exponent (PLE) when fitting a lorentzian to the PSD for knee frequency estimation. Default is `false`. If true, the function will admit Lorentzian's of form ``\frac{A}{1 + (f/a)^b}`` where ``b`` is not confined to -2. 

* `constrained`: Whether or not to use constrained optimization when fitting a lorentzian to the PSD for knee frequency estimation. Default is `false`. If true, the function will use constrained optimization via Optim.jl (using Optimization.jl as a frontend). The lower constraints for amplitude, knee frequency and exponent are 0, freqs[1], 0 respectively. The upper constraints are Inf, freqs[end], 5.0. For optimization, LBFGS method is used. 

* `parallel`: Whether or not to use parallel computation. Default is `false`. If true, the function will use the `OhMyThreads` library to parallelize the computation. 

* `solver`: ("Solver for NonlinearSolve.jl. See https://docs.sciml.ai/NonlinearSolve/stable/solvers/nonlinear_system_solvers/. Defaults to `LevenbergMarquardt`

* `solver_kwargs`: Keyword arguments for NonlinearSolve.solve(). See https://docs.sciml.ai/NonlinearSolve/stable/basics/solve/#solver_options. Defaults to `Dict(:verbose => false)`

## Returns

* `acwresults`: An `ACWResults` object. It has the fields `fs`, `acw_results`, `acwtypes`, `n_lags`, `freqlims`, `acf`, `psd`, `freqs`, `lags`, `x_dim`. You can access these fields as `acwresults.field`. The field `acw_results` contains the ACW results indicated by the input argument `acwtypes` in the same order you specify. Each element of `acw_results` is an array of the same size of your data minus the dimension of time, which will be dropped. See below for details. 

The reason to not return the results directly but return the `ACWResults` object is 1) give access to ACF and PSDs  where the calculations are performed as well as `n_lags` and `freqlims` if the user is using defaults, 2) make plotting easy. You can simply type `acwplot(acwresults)` to plot ACF and PSDs. 

Your primary interest should be the field `acwresults.acw_results`. This is a vector of arrays. Easiest way to explain this is via an example: 

```julia
data = randn(2, 1000, 10) # assume 2 subjects, 1000 time points and 10 trials
fs = 1.0
acwresults = acw(data, fs; acwtypes=[:acw0, :tau, :knee], dims=2)
acw_results = acwresults.acw_results 
```

`acw_results` is a two element vector containing the results with the same order of `acwtypes` as you specify. Since we wrote `:acw0` as the first element and `:tau` as the second element, and `:knee` as the third element, we can extract the results as 

```julia
acw_0 = acw_results[1]
tau = acw_results[2]
knee = acw_results[3]
```

Let's check the dimensionality of these results. Remember that we specified 2 subjects, 1000 time points and 10 trials. The result collapses the dimension of time and gives the result as an 2x10 matrix. 

```julia
size(acw_0) # should be (2, 10)
size(tau) # should be (2, 10)
```

Other fields:

* `fs`: Sampling rate. Floating point number. 

* `acwtypes`: The ACW types you specified. 

* `n_lags`: The number of lags used to fit exponential decay function to the autocorrelation function. See above in the input arguments for details. An integer.

* `freqlims`: Frequency limits to fit a lorentzian to the power spectrum. See above in the input arguments for details. A tuple of floating point numbers. 

* `acf`: Autocorrelation function(s). Has the same size of your data with the time dimension replaced by lag dimension with `n_lags` elements. 

* `psd`: Power spectrum/spectra. Has the same size of your data with the time dimension replaced by frequency dimension with `freqlims` as lowest and highest frequencies. 

* `freqs`: Frequencies corresponding to PSD. 

* `lags`: Lags corresponding to ACF. 

* `x_dim`: The dimension corresponding to lags and frequencies. Used internally in plotting. 

## Plotting

The function `acwplot` can plot power spectra and autocorrelation functions. Currently it supports only two dimensions (for example, subjects x time or trials x time). 

```julia
p = acwplot(acwresults; only_acf=false, only_psd=false, show=true)
```

#### Mandatory Arguments

* `acwresults`: `ACWResults` type obtained by running the function `acw`. 

#### Optional Arguments

* `only_acf` / `only_psd`: Plot only the ACF or only the PSD. Boolean. 

* `show`: Whether to show the plot or only return the variable that contains the plot. 

#### Returns

* `p`: The plot for further modification using the [Plots](https://docs.juliaplots.org/stable/) library. 

