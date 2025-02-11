# [One Timescale with Missing Data (`one_timescale_with_missing_model`)](@id one_timescale_with_missing)

Uses the same syntax as [`one_timescale_model`](one_timescale.md). We refer the user to the documentation of [`one_timescale_model`](one_timescale.md) for details and point out the differences here. 

The generative model is the same as [`one_timescale_model`](one_timescale.md): 

```math
\frac{dx}{dt} = -\frac{x}{\tau} + \xi(t)
```

with timescale ``\tau``. ``\xi(t)`` is white noise with unit variance. The missing data points will be replaced by NaNs as in:

```julia
generated_data[isnan.(your_data)] .= NaN
```

To compute the summary statistic, [`comp_ac_time_missing`](@ref) for ACF and [`comp_psd_lombscargle`](@ref) for PSD is used. Note that PSD is not supported for ADVI method since the [`comp_psd_lombscargle`](@ref) is not autodifferentiable. 

For arguments and examples, see [the documentation for `one_timescale_model`](one_timescale.md). Just replace `one_timescale_model` with `one_timescale_with_missing_model`. 

