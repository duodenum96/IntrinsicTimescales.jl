# Autocorrelation Windows

We finished the [previous section] with a discussion about how a determinstic 
statistic can be influenced by the limitations of our data. In this section, we 
will generalize the problem and discuss various autocorrelation window (ACW) types. 

First, some nomenclature. When we make an analysis on the data, for example, calculate 
event-related potentials, ACWs and so on, we are aiming for an _estimand_. In 
event-related potentials, our estimand is the stereotypical response of the brain 
to some cognitive task. In ACWs, our estimand is intrinsic neural timescales (INTs). 
The ACW we get is not INT per se, it is the _estimate_ of INT. To obtain the estimate, 
we use an _estimator_. The schema below shows the relationship. 

![](assets/practice_2_estimator.drawio.svg)

Our first note about the noise of estimators was the finiteness of the data. We 
noted that as we go along further lags, we have less data points at our hand to calculate 
the correlation values, making the estimate noisier. A first response to the problem 
is to use a different cutoff. Instead of waiting the autocorrelation function to reach 
exactly to 0 thus completely losing the similarity, we can cut it off when it 
reaches 0.5 and say losing half of the similarity. After all, a time-series with a 
longer timescale should take longer to lose half of it. This method is called ACW-50. It is  older than ACW-0. To my knowledge, used first in [Honey et al., 2012](https://pubmed.ncbi.nlm.nih.gov/23083743/). This was a time when the phrase intrinsic neural timescale had 
not been established. The term at that time was temporal receptive windows (TRW). I will 
discuss the evolution of the term more in the [Theory] section. For now, we will make 
simulations from two processes with different timescales and see how well we can distinguish 
their INTs using ACW-50 versus ACW-0. To quickly get many simulations with the same timescale, 
I will set num_trials to 1000. 

```julia
using IntrinsicTimescales # import INT package
using Random 
using Plots # to plot the results
Random.seed!(1) # for replicability

timescale_1 = 1.0
timescale_2 = 3.0
sd = 1.0 # sd of data we'll simulate
dt = 0.001 # Time interval between two time points
duration = 10.0 # 10 seconds of data
num_trials = 1000 # Number of trials

data_1 = generate_ou_process(timescale_1, sd, dt, duration, num_trials)
data_2 = generate_ou_process(timescale_2, sd, dt, duration, num_trials)
println(size(data_1)) # == 30, 1000: 30 trials and 10000 time points
```

To streamline the ACW calculation,  
I will use the `acw` function from INT.jl. This function takes your time series data, 
sampling rate and ACW types you want to calculate and returns the ACW values in the same 
shape of the data. It is best to demonstrate with an example. 

```julia
fs = 1 / dt # sampling rate
acwresults_1 = acw(data_1, fs, acwtypes=[:acw50, :acw0]) 
acwresults_2 = acw(data_2, fs, acwtypes=[:acw50, :acw0])
# Since we used the order [:acw50, :acw0], the first element of results is ACW-50, the second is ACW-0.
acw50_1 = acwresults_1[1]
acw0_1 = acwresults_1[2]
acw50_2 = acwresults_2[1]
acw0_2 = acwresults_2[2]
```

How to quantify the sensitivity of the estimator (to changes in the timescale)? If we claim that 
a time-series with longer timescale should give higher ACW values, the proportion of the INT estimates 
from the longer timescale data that have a smaller ACW value than the median of those from the short 
timescale data or vice versa should give us an idea of how _bad_ an estimator is. Take a look at the code below, 
we will calculate what I described in the previous awful sentence. Hopefully the code is cleaner than my English. Additionally, we will plot histograms to visualize the overlap between estimates. 

```julia
using Printf
median_short_acw50 = median(acw50_1)
median_short_acw0 = median(acw0_1)
median_long_acw50 = median(acw50_2)
median_long_acw0 = median(acw0_2)

bad_acw50_long_timescale = mean(acw50_2 .< median_short_acw50) * 100
bad_acw0_long_timescale = mean(acw0_2 .< median_short_acw0) * 100
bad_acw50_short_timescale = mean(acw50_1 .> median_long_acw50) * 100
bad_acw0_short_timescale = mean(acw0_1 .> median_long_acw0) * 100

# Plot histograms
p1 = histogram(acw50_1, alpha=0.5, label="timescale 1 = $(timescale_1)")
histogram!(p1, acw50_2, alpha=0.5, label="timescale 2 = $(timescale_2)")
# Plot the median since distributions are not normal
vline!(p1, [median_short_acw50, median_long_acw50], linewidth=3, color=:black, label="") 
title!(p1, "ACW-50\n")
# Mad string manipulation
annotate!(p1, 1, 100, 
    (@sprintf("Proportion of \"wrong\" long timescale \nestimates: %.2f%% \n", bad_acw50_long_timescale) * 
    @sprintf("Proportion of \"wrong\" short timescale \nestimates: %.2f%%", bad_acw50_short_timescale), :left))
# ACW-0
p2 = histogram(acw0_1, alpha=0.5, label="timescale 1 = $(timescale_1)")
histogram!(p2, acw0_2, alpha=0.5, label="timescale 2 = $(timescale_2)")

vline!(p2, [median_short_acw0, median_long_acw0], linewidth=3, color=:black, label="")
title!(p2, "ACW-0\n")
annotate!(p2, 2, 175, 
    (@sprintf("Proportion of \"wrong\" long timescale \nestimates: %.2f%% \n", bad_acw0_long_timescale) * 
    @sprintf("Proportion of \"wrong\" short timescale \nestimates: %.2f%%", bad_acw0_short_timescale), :left),
    textfont=font(24))
plot(p1, p2, size=(1600, 800))
```

![](assets/practice_2_1.svg)

_to be continued_