# Model-Free Timescale Estimation

Performed via the function `acw` in INT.jl. 

```julia
acwresults = acw(data, fs; acwtypes=possible_acwtypes, n_lags=nothing, freqlims=nothing, dims=ndims(data))
```

Mandatory arguments: 
`data`: Your data as a vector or n-dimensional array. If it is n-dimensional, by default, the 
dimension of time is assumed to be the last dimension. For example, if you have a 2D array where 
rows are subjects and columns are time points, `acw` function will correctly assume that 