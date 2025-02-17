# [One Timescale and Oscillation with Missing Data (`one_timescale_and_osc_with_missing_model`)](@id one_timescale_and_osc_with_missing)

Uses the same syntax as [`one_timescale_model`](one_timescale.md) and has the same implementation details (i.e. three priors and three results) as [`one_timescale_and_osc`](one_timescale_and_osc.md). We refer the users to the respective documentations. The only difference of `one_timescale_and_osc_with_missing_model` from `one_timescale_and_osc` is that missing data points is replaced with NaNs in the generative model, as in [`one_timescale_with_missing_model`](one_timescale_with_missing.md). 

