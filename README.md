[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://duodenum96.github.io/IntrinsicTimescales.jl/stable/home)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://duodenum96.github.io/IntrinsicTimescales.jl/dev/home)
[![build](https://github.com/duodenum96/.jl/workflows/CI/badge.svg)](https://github.com/duodenum96/IntrinsicTimescales.jl/actions?query=workflow%3ACI)
# IntrinsicTimescales.jl

IntrinsicTimescales.jl is a software package for estimating Intrinsic Neural Timescales (INTs) from time-series data. It uses model-free methods (ACW-50, ACW-0, fitting an exponential decay function etc.) and simulation-based methods (adaptive approximate Bayesian computation: aABC, automatic differentiation variational inference: ADVI) to estimate INTs.

The documentation is available [here](https://duodenum96.github.io/IntrinsicTimescales.jl/dev/home/).

## Installation

This package is written in Julia. If you do not have Julia installed, you can install it from [here](https://julialang.org/downloads/). Once you have Julia installed, you can install IntrinsicTimescales.jl by running the following command in the Julia REPL:

```julia
using Pkg
Pkg.add("IntrinsicTimescales")
```
Soon, there will also be a Python wrapper called [INTpy](https://github.com/duodenum96/INTpy), which will allow you to use IntrinsicTimescales.jl from Python. 

## Quickstart

IntrinsicTimescales.jl uses two ways to estimate INTs: model-free methods and simulation-based inference. Model-free methods include ACW-50, ACW-0, ACW-e, area under the curve (AUC), decay rate of an exponential fit to ACF and knee freqency of a lorentzian fit to PSD. Simulation-based methods are based on [Zeraati et al. (2022)](https://www.nature.com/articles/s43588-022-00214-3) paper and do parameter estimation by assuming the data came from an Ornstein-Uhlenbeck process. For estimation, in addition to the adaptive approximate Bayesian computation (aABC) method used in [Zeraati et al. (2022)](https://www.nature.com/articles/s43588-022-00214-3), we also present automatic differentiation variational inference (ADVI). Additionally, we adapt the aABC method with adaptive choice of epsilon. See documentation for details.

For model-free methods, simply use the `acw` function.

```julia
using IntrinsicTimescales
acwresults = acw(data, fs; acwtypes = [:acw0, :acw50, :acweuler, :auc, :tau, :knee]), dims=ndims(data))
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

These functions are highly customizable, see the [documentation](https://duodenum96.github.io/IntrinsicTimescales.jl/dev/home/) for details. 

## Getting Help and Making Contributions

Questions and contributions are welcome. Use the [issues section of our github page](https://github.com/duodenum96/IntrinsicTimescales.jl/issues) to report bugs, make feature requests, ask questions or tackle the issues by making pull requests. 

## Want to learn more?

[Kindly read the fine manual (RTFM).](https://duodenum96.github.io/IntrinsicTimescales.jl/dev/home/)

## License

This project is licensed under the MIT License - see the LICENSE file for details. Note that certain dependencies (such as FFTW) may not have MIT licenses. 

<!-- Tidyverse lifecycle badges, see https://www.tidyverse.org/lifecycle/ Uncomment or delete as needed -->
<!-- ![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg) -->
<!-- ![lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg) -->
<!-- ![lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg) -->
<!-- ![lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg) -->
<!-- ![lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)  -->
<!-- ![lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->

<!-- travis-ci.com badge, uncomment or delete as needed, depending on whether you are using that service. -->
<!-- [![Build Status](https://travis-ci.com/duodenum96/IntrinsicTimescales.jl.svg?branch=master)](https://travis-ci.com/duodenum96/IntrinsicTimescales.jl) -->
<!-- NOTE: Codecov.io badge now depends on the token, copy from their site after setting up -->
<!-- Documentation -- uncomment or delete as needed -->

<!-- Aqua badge, see test/runtests.jl -->
<!-- [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) -->
