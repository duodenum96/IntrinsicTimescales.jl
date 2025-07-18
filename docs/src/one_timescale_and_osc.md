# [One Timescale and Oscillation Model (`one_timescale_and_osc_model`)](@id one_timescale_and_osc)

Uses the same syntax as [`one_timescale_model`](one_timescale.md). We refer the user to the documentation of [`one_timescale_model`](one_timescale.md) for details and point out the differences here. 

The generative model: 

```math
\frac{dy}{dt} = -\frac{y}{\tau} + \xi(t) \\
\\
\\
x(t) = \sqrt{a}y(t) + \sqrt{1-a} sin(2 \pi f t + \phi)
```

where $f$ is the frequency, $a$ is the weight of the Ornstein-Uhlenbeck (OU) process and $ \phi $ is a random number drawn from a normal distribution to reflect a random phase offset for each trial. Note that now we need to fit three parameters: $ \tau $ for timescale, $f$ for the oscillation frequency and $ a $ for how strong the oscillations are (with a smaller $a$ indicating larger oscillations). Note that $a$ is bounded between 0 and 1. Similarly, the maximum a posteriori estimates (MAP) also has three elements: one for each prior. Due to the three parameters needed, the fitting is more difficult compared to  `one_timescale_model`. 

If the user wishes to set the priors, they need to specify a prior for each of the parameters. The ordering is 1) the prior for timescale, 2) the prior for frequency second and 3) the prior for the coefficient. An example:

```julia
using Distributions, IntrinsicTimescales

data_mean = 0.0 # desired mean
data_sd = 1.0 # desired sd
duration = 10.0 # duration of data
n_trials = 10 # How many trials
fs = 500.0 # Sampling rate
timescale = 0.05 # 50 ms
oscillation_freq = 10.0 # 10 Hz alpha oscillation
coefficient = 0.95
theta = [timescale, oscillation_freq, coefficient] # vector of parameters

data = generate_ou_with_oscillation(theta, 1/fs, duration, n_trials, data_mean, data_sd)


priors = [
        Normal(0.1, 0.1),    # a prior for a 0.1 second timescale with an uncertainty of 0.1
        Normal(10.0, 5.0),   # 10 Hz frequency with uncertainty of 5 Hz
        Uniform(0.0, 1.0)    # Uniform distribution for coefficient
    ]

time = (1/fs):(1/fs):duration
model = one_timescale_and_osc_model(data, time, :abc, summary_method=:acf, prior=priors)
results = int_fit(model)
int = results.MAP[1]  # max a posterori for INT
freq = results.MAP[2] # for frequency
coef = results.MAP[3] # for coefficient
```

If the user does not specify a prior or sets `prior="informed_prior"`, IntrinsicTimescales.jl generates priors from data. The prior for the coefficient in this case is `Uniform(0.0, 1.0)`. For `summary_method=:acf`, the timescale prior is an exponential decay fit to the ACF from data whereas `summary_method=:psd` fits a Lorentzian function to the PSD from data, obtains the knee frequency and estimates the timescale from it as in `one_timescale_model`. The prior for the frequency is obtained with first fitting a Lorentzian to the PSD, then subtracting the lorentzian to eliminate aperiodic component as in [FOOOF] and finally obtains the peak frequency with [`find_oscillation_peak`](@ref). 

Similarly, the argument `combine_distance=true` not only calculates the RMSE between PSDs or ACFs, but also combines that distance with the RMSE between timescale and frequency estimates between the model and data. 

The other arguments are the same as `one_timescale_model`. We refer the reader to [that section of the documentation](one_timescale.md) for details. 
