# Dealing with Oscillatory Artifacts

So far, we only dealt with autocorrelation functions (ACFs) that are of exponential decay type. This is fine for fMRI data but can be problematic for EEG/MEG data. To demonstrate, I'll use the function [`generate_ou_with_oscillation`](@ref). This function adds an oscillation on top of an Ornstein-Uhlenbeck process. It takes three parameters for its generative model: timescale, oscillation frequency and coefficient for OU process (meaning higher the coefficient, lower the oscillatory artifact). The coefficient is bounded between 0 and 1. If you try to give it a coefficient that is greater than 1 or smaller than 0, it will change it to 1 and 0 respectively. It also takes the desired mean and sd for data. This is required for Bayesian estimation of timescales which will be the topic of next tutorial. Consider the following code. I'll simulate two time-series, with and without oscillatory artifacts and calculate ACW-e which I introduced in the previous section. 

```julia
using IntrinsicTimescales, Plots, Random, Statistics
Random.seed!(666) # for reproducibility
fs = 1000.0 # 1000 Hz sampling rate
dt = 1.0 / fs
duration = 10 # 10 seconds of data
num_trials = 10
data_mean = 0.0 # desired mean
data_sd = 1.0 # desired sd

timescale = 0.3 # 300 ms
oscillation_freq = 10.0 # 10 Hz alpha oscillation
coefficient = 0.95
theta = [timescale, oscillation_freq, coefficient] # vector of parameters

data_osc = generate_ou_with_oscillation(theta, dt, duration, num_trials, data_mean, data_sd)
data = generate_ou_process(timescale, data_sd, dt, duration, num_trials)
acwresults_osc = acw(data_osc, fs; acwtypes=:acweuler)
acwresults = acw(data, fs; acwtypes=:acweuler)
println(mean(acwresults_osc.acw_results))
# 0.087
println(mean(acwresults.acw_results))
# 0.3075
p1 = acwplot(acwresults_osc)
title!(p1, "ACF with oscillatory component")
p2 = acwplot(acwresults)
title!(p2, "ACF")
plot!(p1, p2)
```

The ACW calculated from the simulation with the oscillatory component is terribly off. A good question is why do we see oscillations in the ACF (note the wiggles)? Remember that ACF is calculating the correlation of your data with your data shifted by a certain lag. If you consider a perfect oscillation, its correlation with itself will fluctuate. Whenever the peaks and troughs of oscillation correspond to peaks and troughs of the shifted oscillation (which will happen periodically, when you shift the oscillation just right enough so that it matches with itself), it will nicely correlate with itself. If you shift it half the period of oscillation, so that the peaks of the oscillation will match the troughs of the shifted oscillation, it will have a negative autocorrelation. (Try to draw this on your notebook to get a clearer picture). As a result, the ACF of a perfect oscillation is another oscillation. But oscillation here is also coupled with the OU process. Hence the exponential decay + oscillation type of ACF. 

We need to find a way to decouple the oscillation from the OU process. There is a mathematical technology for this, called the Fourier transform. Essentially it is the correlation of a signal with an oscillation. Let's write down the math. 

```math

```




```julia
acf = comp_ac_fft(data)
psd, freqs = comp_psd(data, fs)
plot(freqs, psd', scale=:log10)
xlims!((1.0, 50.0))
plot(acf')
```

