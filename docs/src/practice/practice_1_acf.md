# Building the Autocorrelation Function

Data is noisy. Each time point has a random deviation. It is meaningless to ask anything about a single time point. However, certain statistical properties of random data are not random. For example, if I flip a coin 1000 times it is meaningless to ask whether the 348th flip will be heads or tails but on average, half the time I will get heads and half the time I will get tails. _Correlation time_ is a statistical property of time-series data. It is not random: you can get many different random time series with the same correlation time. Correlation time measures how long does it take for a signal to lose similarity to itself. Why should we care? It is the basis of intrinsic neural timescales (INTs) and since you are here, I am assuming that you care about INTs. I'll explain more in the [Theory](../theory/theory.md) sections. For now, let's just assume that it matters and learn how to calculate it. 

To quantify the similarity between two things, we can use correlation. The higher the correlation, more similar two things are. The assumption that something loses similarity with itself implies that initially there was a similarity but over time we lost it. To quantify similarity of something with itself at a later time, we can calculate the correlation between that thing and that thing pushed forward in time. It is easier to see this with a figure. 

![](assets/practice_acf_1_drawio.svg)

We took the time series x and shifted it forward in time by an amount `` \Delta``t. Then we need to take the correlation between them. To take a correlation between two things, you need to have equal number of data points in each of them. This is due to the definition of correlation, correlation is the average value of multiplication normalized by variance. You need to multiply corresponding data points. Take a look at the code example below. Throughout the documentation, there will be many code examples. I encourage you to run them on your computer and play around with them. Even better, take a pen and piece of paper and do the calculation below yourself. There is no better way to train intuition other than grinding your way through a calculation but I digress. Here is the code:

```julia
using Statistics # Import Statistics package for cor function
x1 = [-1, 0, 1] # example data
x2 = [2, -2, 0]
variance_x1 = sqrt(sum(x1 .^ 2) / 3) # Calculate variance of each dataset
variance_x2 = sqrt(sum(x2 .^ 2) / 3)
# Covariance is the average value of multiplication
covariance_x1_x2 = (x1[1]*x2[1] + x1[2]*x2[2] + x1[3]*x2[3]) / 3
# Correlation is normalized covariance
correlation_x1_x2 = covariance_x1_x2 / (variance_x1 * variance_x2)
isapprox(correlation_x1_x2, cor(x1, x2)) # Compare with cor function from Statistics package
```

This looks basic, but makes an important point. As you go forward in time, you need to match the time points in your time series and shifted version of it. In the figure above, the only usable part is the part indicated in black vertical lines. This means as we shift further in time, we have less time points at our disposal and our correlation results are less reliable. We will return back to this point later. 

We took a time-series, shifted it by an amount ``\Delta``t, calculated the correlation and if the result is not zero, then we can say that the time series still hasn't lost similarity to itself in ``\Delta``t amount of time. Take a moment to ponder about this sentence. We are insinuating that there is such a ``\Delta``t where the correlation is zero, or close to zero and this is the time it takes for a signal to lose similarity with itself. This is our INT. 

Then a good strategy to calculate INT is simply calculating the correlation at various ``\Delta``t values and detecting which ``\Delta``t is the time where we lose correlation. Let's code this. We'll use the function [`generate_ou_process`](@ref) from the IntrinsicTimescales.jl package. This function simulates time series with a known timescale. I'll explain more about what it is doing in [Theory](../theory/theory.md) section. For now, just know that this exists and is a good toy to play with. In IntrinsicTimescales.jl package, we have more optimized ways to do the operation I'll write below. I am doing this below explicitly and in detail so that we know exactly what we are doing when we compute these things. 

```julia
using IntrinsicTimescales # import INT package
using Random 
using Plots # to plot the results
Random.seed!(1) # for replicability

timescale = 1.0
sd = 1.0 # sd of data we'll simulate
dt = 0.001 # Time interval between two time points
duration = 10.0 # 10 seconds of data
num_trials = 1 # Number of trials

data = generate_ou_process(timescale, sd, dt, duration, num_trials)
data = data[:] # Go from a (1, time) matrix to (time) vector
```

The resulting `data` from `generate_ou_process` is a matrix where rows are different trials and columns are time points. In order to simplify the code below, I do the operation `data = data[:]` to turn it into a one dimensional vector. 

The next step is shifting forward in time and correlating on this data. Look at the code below, take a piece of pen and paper and explicitly write down the indexing operations for different values of ``\Delta``t to get a sense of how we are implementing this. Essentially, we are finding the indices corresponding to the data between the black vertical lines shown in the figure above. 

```julia
n_timepoints = length(data)
n_lags = 4000 # Calculate the first 4000 lags.
correlation_results = zeros(n_lags) # Initialize empty vector to fill the results
# Start from no shifting (0) and end at number of time points - 1. 
lags = 0:(n_lags-1)
for DeltaT in lags
    # Get the indices for the data in vertical lines
    indices_data = (DeltaT+1):n_timepoints
    indices_shifted_data = 1:(n_timepoints - DeltaT)
    correlation_results[DeltaT+1] = cor(data[indices_data], data[indices_shifted_data])
end
plot(lags, correlation_results, label="") 
hline!([0], color=:black, label="") # Indicate the zero point of correlations
```

![](assets/intro_2.svg)

This is called an _autocorrelation function (ACF)_. On x axis, we have lags. One lag means we shifted one of the time series by one data point. On y axis, we plot the correlation values. Note that it starts from 1. Because when lag is zero, we did not shift any time series. We are correlating a time series with exactly itself and the correlation between one thing and itself is simply one. As we expected, the ACF decays as we shift lags. We can identify the lag where the correlation reaches zero. This is the first estimate of our timescale. This measure is called _ACW-0_ which stands for _autocorrelation window-0_. It was first used by [Mehrshad Golesorkhi in his 2021 paper](https://www.nature.com/articles/s42003-021-01785-z) and he found that ACW-0 differentiates brain regions better than previously used methods. Let's calculate the ACW-0 and indicate it in the plot with a vertical red line. 

```julia
acw_0 = findfirst(correlation_results .< 0)
plot(correlation_results, xlabel="Lags", ylabel="Correlation", label="")
hline!([0], color=:black, label="")
vline!([acw_0], color=:red, label="ACW-0")
```

![](assets/intro_3.svg)

So our work is done, right? We started with 1) the definition that INT is the time it takes for a time-series to lose its similarity with itself, 2) operationalized similarity with correlation, 3) operationalized similarity with itself as correlation with itself shifted some time lags and 4) identified the INT as the number of time lags required to lose similarity. There is one problem. Remember the problem of number of time points we talked about above. As we go further in lags, we have less and less number of data points to calculate the correlation, the portion inside vertical black lines is getting smaller and smaller. If we do not have enough number of data points to calculate ACW-0, then we will get a noisy estimate. 

Let's try to see how big of a problem this is. Below, we will simulate the time-series again and again and overlay plots of ACFs. In a different panel, we'll do a histogram of ACW-0 values. To calculate ACF, we will use the function [`comp_ac_fft`](@ref) from  IntrinsicTimescales.jl package. This function is faster and uses a different technique to calculate ACF which I'll explain in the [next section]. For now, it should suffice to know that it takes the data and optionally the number of lags we want as input and gives back the ACF. If number of lags is not specified, it goes through all possible lags. To get the ACW-0 from the ACF, we'll use [`acw0`](../acw.md) function which takes lags and ACF as input and gives ACW-0 value. 

```julia
acw0_results = [] # Initialize empty vectors to hold the results
acfs = []
n_simulations = 10
for _ in 1:n_simulations
    data = generate_ou_process(timescale, sd, dt, duration, num_trials)[:]
    acf = comp_ac_fft(data; n_lags=n_lags)
    i_acw0 = acw0(lags, acf)
    push!(acw0_results, i_acw0) # Same as .append method in python
    push!(acfs, acf)
end
p1 = plot(lags, acfs, xlabel="Lags", ylabel="Correlation", 
          label="", title="ACF", alpha=0.5)
hline!([0], color=:black, label="")

p2 = histogram(acw0_results, xlabel="ACW-0", ylabel="Count",
               label="", title="Distribution of ACW-0")

# Combine the plots side by side
plot(p1, p2, layout=(1,2))
```

![](assets/intro_4.svg)

What's going on  here? We simulated the same process 10 times and each time we got a different result. All simulations had the same timescale, which we set as 1.0 above. Why did we get different results? Didn't we start by saying that even the data is random, statistical properties of it are not? That we can flip a coin 1000 times and on average half the time it will be heads and half the time it will be tails? Well, not quite. We said that _on average_, half the time it will be heads and the other half, it will be tails. Let's define an experiment as flipping the coin 1000 times. If you do this experiment once and look at the results, perhaps it will be 498 heads and 502 tails. Then do the experiment again, it will maybe give you 505 heads and 495 tails. You do the experiment again and again and keep track of the results. Then if you average over experiments, you'll see that there are 500 heads and 500 tails in the end. You can do a mini version of this experiment at home with 10 coin flips. The more experiments you do, the better the results will be. 

Here is the central insight: When we calculate ACW-0 from limited data, we are not doing a perfect calculation. We are making an estimation. Based on the data we know, this is the timescale we think. And estimations are noisy. The noisiness of the estimation depends on the properties of data. The more number of data points we have, the better the estimations are. This is why I stressed that as we calculate ACF in later and later lags, our estimations become less and less reliable simply because we have less number of data points at our disposal. To see it clearly, look at the figure in the left panel and observe that at earlier lags, the variance between ACF estimates are low and it progressively increases as you go along later lags. Feel free to change the parameters `dt`,  `timescale` and `duration` to see how they change results. 

This is why it is crucial to not only know your research problem, be it cognitive or basic neuroscience, but also the estimators you use to tackle the problem. How noisy are they? How much they are vulnerable to the number of data points? Are there other things in the data that might bias the results? Just because you are getting a number out of some algorithm does not mean that number has any meaning. It is the responsibility of the researcher, _you_ to make sure your numbers make sense. 

In the [next section](practice_2_acw.md), we will explore various kinds of _autocorrelation windows_, their motivation and how they address the bias. 