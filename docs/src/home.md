# IntrinsicTimescales.jl Documentation

Welcome to the documentation of IntrinsicTimescales.jl. IntrinsicTimescales.jl is a software package for estimating Intrinsic Neural Timescales (INTs) from time-series data. It uses model-free methods (ACW-50, ACW-0, fitting an exponential decay function etc.) and simulation-based methods (adaptive approximate Bayesian computation: aABC, currently experimental automatic differentiation variational inference: ADVI) to estimate INTs.

## Installation

This package is written in Julia. If you do not have Julia installed, you can install it from [here](https://julialang.org/downloads/). Once you have Julia installed, you can install IntrinsicTimescales.jl by running the following command in the Julia REPL:

```julia
using Pkg
Pkg.add("IntrinsicTimescales")
```
Soon, there will also be a Python wrapper called [INTpy](https://github.com/duodenum96/INTpy), which will allow you to use IntrinsicTimescales.jl from Python. 

## Quickstart

IntrinsicTimescales.jl uses two ways to estimate INTs: model-free methods and simulation-based inference. Model-free methods include ACW-50, ACW-0, ACW-e, decay rate of an exponential fit to ACF and knee freqency of a lorentzian fit to PSD. Simulation-based methods are based on [Zeraati et al. (2022)](https://www.nature.com/articles/s43588-022-00214-3) paper and do parameter estimation by assuming the data came from an Ornstein-Uhlenbeck process. For estimation, in addition to the aABC method used in [Zeraati et al. (2022)](https://www.nature.com/articles/s43588-022-00214-3), we also present ADVI. 

For model-free methods, simply use 

```julia
using IntrinsicTimescales

data = randn(10, 5000) # Data in the form of (trials x time) 
fs = 100.0 # Sampling frequency

acwresults = acw(data, fs; acwtypes = [:acw0, :acw50, :acweuler, :auc, :tau, :knee], dims=ndims(data))
# or even simpler:
acwresults = acw(data, fs)
```

where `fs` is sampling frequency, optional parameters `acwtypes` is a vector of 
symbols (indicated with `:`) telling which methods to use and `dims` is indicating the dimension of time in your array (by default, the last dimension). The resulting `acwresults` gives the results in the same order of `acwtypes`. 

For simulation based methods, pick one of the `one_timescale_model`, `one_timescale_with_missing_model`, `one_timescale_and_osc_model` and `one_timescale_and_osc_with_missing_model` functions. These models correspond to different generative models depending on whether there is an oscillation or not. For each generative model, there are with or without missing variants which use different ways to calculate ACF and PSD. Once you pick the model, the syntax is 

```julia
# Simulate some data
using Random
timescale = 0.3 # true timescales
variance = 1.0 # variance of data
duration = 10.0 # duration of data
n_trials = 2 # How many trials
fs = 500.0 # Sampling rate
rng = Xoshiro(123); deq_seed = 123 # For reproducibility
data = generate_ou_process(timescale, variance, 1/fs, duration, n_trials, rng=rng, deq_seed=deq_seed) # Data in the form of (trials x time)
time = (1/fs):(1/fs):duration # Vector of time points
model = one_timescale_model(data, time, :abc)
result = int_fit(model)
```

or 

```julia
model = one_timescale_model(data, time, :advi)
result = int_fit(model)
```

These functions are highly customizable, see the page [Simulation Based Timescale Estimation](simbasedinference.md). 

## Organization of the package

The diagram below shows the rough organization of the package:

![](assets/diagram.svg)

## Where to go from here?

This documentation is divided into three parts. [Explanation](practice/practice_intro.md) part is divided into theory and practice sections. The first section is [Practice](practice/practice_intro.md). It is usually easier to understand something after you do it, therefore, I placed the practice section before theory. In [Practice](practice/practice_intro.md), we carefully build our way towards estimating INTs by starting from the autocorrelation function and slowly proceeding to more and more advanced methods. If you never calculated INTs before, this is where you should start. The second part is [Theory](theory/theory.md). This part aims to delve into the history of INT research, what it means in the brain and what it is good for with a particular emphasis on theoretical research, summarizing the cutting edge in this frontier. It is especially useful for researchers working on INT itself. Right now, it is in construction. I will deploy it as soon as it is ready. 

The third part is the [Reference](acw.md). This part contains the [Implementation](acw.md) and [API](index.md). The [API](index.md) is an exhaustive list of functions and their signatures in the package. It is boring and most of the functions are not intended for end-user (you). The implementation part documents [model-free](acw.md) and [simulation-based](simbasedinference.md) methods that are used in the package, with the full function signatures. This part should serve as the reference for the user. If you are already familiar with INTs and want to see how to use this package, you can start here. 

Finally, the [Tutorial](tutorial/tutorial_1_acw.md) contains practical considerations for picking which INT metric to use in which situation, as well as [using the package with MNE-Python](tutorial/tutorial_2_mne.md) and [with FieldTrip](tutorial/tutorial_3_ft.md) packages. 

## Getting Help and Making Contributions

Questions and contributions are welcome. Use the [issues section of our github page](https://github.com/duodenum96/IntrinsicTimescales.jl/issues) to report bugs and make feature requests and ask questions. Please see [Contributing Guidelines](contributing.md) before contributing. 

## Statement of Need

Intrinsic neural timescales (INTs) were found to be an important metric to probe the brain dynamics and function. On the neuroscientific side, INTs were found to follow the large-scale gradients in the cortex ranging from uni to transmodal areas including local and long-range excitation and proxies of myelination. From a cognitive science perspective, INTs were found to be related to reward, behavior, self, consciousness among others. Proper estimation of INTs to make sure the estimates are not affected by limited data, missingness of the data and oscillatory artifacts is crucial. While several methods exist for estimating INTs, there is a lack of standardized, open-source tools that implement both traditional model-free approaches and modern Bayesian estimation techniques. Existing software solutions are often limited to specific estimation methods, lack proper uncertainty quantification, or are not optimized for large-scale neuroimaging data.

IntrinsicTimescales.jl addresses these limitations by providing a comprehensive, high-performance toolbox for INT estimation. The package implements both established model-free methods and novel Bayesian approaches, allowing researchers to compare and validate results across different methodologies with a simple API. Its implementation in Julia ensures computational efficiency, crucial for analyzing large neuroimaging datasets. The package's modular design facilitates easy extension and integration with existing neuroimaging workflows, while its rigorous testing and documentation make it accessible to researchers across different levels of programming expertise.

## About

This package is developed by me, [Yasir Çatal](https://github.com/duodenum96) during my PhD. I got [nerdsniped](https://xkcd.com/356/) by [Zeraati et al., 2021](https://www.nature.com/articles/s43588-022-00214-3) paper and started writing the package. the rest evolved from the simple motivation of reimplementing [abcTau](https://github.com/roxana-zeraati/abcTau) in Julia with various performance optimizations. 

I am doing my PhD on INTs and [our lab](https://www.georgnorthoff.com) is specialized on the topic. As a result, I had many conversations with [almost every member](https://www.georgnorthoff.com/researchers) of our lab about INTs. I designed this documentation while keeping those conversations in mind. My goal was not only to document the package, but also to build up the knowledge to grasp the concept of INTs especially for new researchers starting their journey and active researchers in the trenches if they wish to brush up their basics. 

## Citations

See [Citations](citations.md) to see the papers you can cite specific to methods. 