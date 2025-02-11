# INT.jl Documentation

Welcome to the documentation of INT.jl. INT.jl is a software package for estimating Intrinsic Neural Timescales (INTs) from time-series data. It uses model-free methods (ACW-50, ACW-0, fitting an exponential decay function etc.) and simulation-based methods (ABC, ADVI) to estimate INTs.

## Installation

This package is written in Julia. If you do not have Julia installed, you can install it from [here](https://julialang.org/downloads/). Once you have Julia installed, you can install INT.jl by running the following command in the Julia REPL:

```julia
using Pkg
Pkg.add("INT")
```
Soon, there will also be a Python wrapper called [INTpy](https://github.com/duodenum96/INTpy), which will allow you to use INT.jl from Python. 

## Quickstart

INT.jl uses two ways to estimate INTs: model-free methods and simulation-based inference. Model-free methods include ACW-50, ACW-0, ACW-e, decay rate of an exponential fit to ACF and knee freqency of a lorentzian fit to PSD. Simulation-based methods are based on [Zeraati et al. (2022)](https://www.nature.com/articles/s43588-022-00214-3) paper and do parameter estimation by assuming the data came from an Ornstein-Uhlenbeck process. For estimation, in addition to the aABC method used in [Zeraati et al. (2022)](https://www.nature.com/articles/s43588-022-00214-3), we also present ADVI. 

For model-free methods, simply use 

```julia
using INT
acwresults = acw(data, fs; acwtypes = [:acw0, :acw50, :acweuler, :tau, :knee]), dims=ndims(data))
# or even simpler:
acwresults = acw(data, fs)
```

where `fs` is sampling frequency, optional parameters `acwtypes` is a vector of 
symbols (indicated with `:`) telling which methods to use and `dims` is indicating the dimension of time in your array (by default, the last dimension). The resulting `acwresults` gives the results in the same order of `acwtypes`. 

For simulation based methods, pick one of the `one_timescale_model`, `one_timescale_with_missing_model`, `one_timescale_and_osc_model` and `one_timescale_and_osc_with_missing_model` functions. These models correspond to different generative models depending on whether there is an oscillation or not. For each generative model, there are with or without missing variants which use different ways to calculate ACF and PSD. Once you pick the model, the syntax is 

```julia
model = one_timescale_model(data, time, :abc)
result = solve(model)
```

or 

```julia
model = one_timescale_model(data, time, :advi)
result = solve(model)
```

These functions are highly customizable, see the page [Simulation Based Timescale Estimation](simbasedinference.md). 

## Where to go from here?

This documentation is divided in four parts. The fourth part [API](index.md) is an exhaustive list of functions and their signatures in the package. It is boring. A better place to start is the third part, Implementation. This part documents [model-free](acw.md) and [simulation-based](simbasedinference.md) methods that are used in the package, with the full function signatures. If you are already familiar with calculating INTs and just want to start using the package, this is the right place. 

The remaining two parts are to understand the motivation to use various methods for calculating INTs and the motivation to calculate INTs (i.e., practice and theory). The first part is [Practice](practice/practice_intro.md). It is usually easier to understand something after you do it, therefore, I placed the practice section before theory. In [Practice](practice/practice_intro.md), we carefully build our way towards estimating INTs by starting from the autocorrelation function and slowly proceeding to more and more advanced methods. The second part is [Theory](theory/theory.md). This part delves into the history of INT research, what it means in the brain and what it is good for with a particular emphasis on theoretical research, summarizing the cutting edge in this frontier. It is especially useful for researchers working on INT itself. 

## Getting Help and Making Contributions

Questions and contributions are welcome. Use the [issues section of our github page](https://github.com/duodenum96/INT.jl/issues) to report bugs, make feature requests, ask questions or tackle the issues by making pull requests. 

## About

This package is developed by me, [Yasir Çatal](https://github.com/duodenum96) during my PhD. I got [nerdsniped](https://xkcd.com/356/) by [Zeraati et al., 2021](https://www.nature.com/articles/s43588-022-00214-3) paper and started writing the package. the rest evolved from the simple motivation of reimplementing [abcTau](https://github.com/roxana-zeraati/abcTau) in Julia with various performance optimizations. 

I am doing my PhD on INTs and [our lab](https://www.georgnorthoff.com) is specialized on the topic. As a result, I had many conversations with [almost every member](https://www.georgnorthoff.com/researchers) of our lab about INTs. I designed this documentation while keeping those conversations in mind. My goal was not only to document the package, but also to build up the knowledge to grasp the concept of INTs especially for new researchers starting their journey and active researchers in the trenches if they wish to brush up their basics. 

