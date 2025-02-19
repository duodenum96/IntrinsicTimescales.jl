using Plots

colors = palette(:Catppuccin_mocha)[[4, 5, 7, 9, 3, 10, 13]]

# Figure 1
n = 1000
offset = 500
time_x = 1:n
time_x2 = ((1+offset):(n+offset))
x = 0.1randn(n) + sin.(time_x*2*pi*0.01)
plot(time_x, x .+ 2, linewidth=2, label="x(t)", palette=colors)
plot!(time_x2, x, linewidth=2, label="x(t+Δt)")
vline!([offset, time_x[end]], color=:black, linewidth=3, label="")
plot!(framestyle=:none, grid=false, ticks=false)
# plot!(background_color=:transparent)
plot!(legendfont=font(12), legend_frame=:none, foreground_color_legend = nothing)

savefig("docs/src/practice/assets/intro_1.svg")

# Figure 2
using IntrinsicTimescales # import INT package
using Random 
using Statistics
Random.seed!(1) # for replicability

timescale = 1.0
sd = 1.0 # sd of data we'll simulate
dt = 0.001 # Time interval between two time points
duration = 10.0 # 10 seconds of data
num_trials = 1 # Number of trials

data = generate_ou_process(timescale, sd, dt, duration, num_trials)
data = data[:]

n_timepoints = length(data)
n_lags = 4000 # Calculate the first 2000 lags.
correlation_results = zeros(n_lags) # Initialize empty vector to fill the results
lags = 0:(n_lags-1)
# Start from no shifting (0) and end at number of time points - 1. 
for DeltaT in 0:(n_lags-1) 
    # Get the indices for the data in vertical lines
    indices_data = (DeltaT+1):n_timepoints
    indices_shifted_data = 1:(n_timepoints - DeltaT)
    correlation_results[DeltaT+1] = cor(data[indices_data], data[indices_shifted_data])
end
plot(correlation_results, label="", linewidth=3, palette=colors, 
    color=colors[5], yticks=[-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
hline!([0], color=:black, linewidth=2, label="") # Indicate the zero point of correlations
savefig("docs/src/practice/assets/intro_2.svg")

# Figure 3
acw_0 = findfirst(correlation_results .< 0)
plot(correlation_results, xlabel="Lags", ylabel="Correlation", label="", 
    color=colors[5], linewidth=3, palette=colors,
    yticks=[-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
hline!([0], color=:black, label="")
vline!([acw_0], color=:red, label="ACW-0")
savefig("docs/src/practice/assets/intro_3.svg")


# Figure 4
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
          label="", title="ACF", alpha=0.5, palette=colors, linewidth=3, 
          yticks=[-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
hline!([0], color=:black, label="", linewidth=3)

p2 = histogram(acw0_results, xlabel="ACW-0", ylabel="Count",
               label="", title="Distribution of ACW-0", palette=colors)

# Combine the plots side by side
plot(p1, p2, layout=(1,2))
savefig("docs/src/practice/assets/intro_4.svg")

# %% -------------------------------------------------------
# Practice 2 - ACW Figures

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

fs = 1 / dt # sampling rate
acwresults_1 = acw(data_1, fs, acwtypes=[:acw50, :acw0]) 
acwresults_2 = acw(data_2, fs, acwtypes=[:acw50, :acw0])
# Since we used the order [:acw50, :acw0], the first element of results is ACW-50, the second is ACW-0.
acw50_1 = acwresults_1.acw_results[1]
acw0_1 = acwresults_1.acw_results[2]
acw50_2 = acwresults_2.acw_results[1]
acw0_2 = acwresults_2.acw_results[2]
using Printf

bad_acw50_timescale = mean(acw50_2 .< acw50_1) * 100
bad_acw0_timescale = mean(acw0_2 .< acw0_1) * 100

# Plot histograms
p1 = histogram(acw50_1, # First element is ACW-50, second is ACW-0, in the same order as acwtypes variable
    alpha=0.5, label="timescale 1 = $(timescale_1)",
    xtickfontsize=16, ytickfontsize=16, palette=colors, color=colors[5])
histogram!(p1, acw50_2, alpha=0.5, label="timescale 2 = $(timescale_2)", color=colors[4])
# Plot the median since distributions are not normal
vline!(p1, [median(acw50_1), median(acw50_2)], linewidth=3, color=:black, label="") 
title!(p1, "ACW-50\n", titlefontsize=24)

annotate!(p1, 1.0, 100, 
    (@sprintf("Proportion of \"wrong\" timescale \nestimates: %.2f%% \n", bad_acw50_timescale)), textfont=font(24), :left)

plot!(p1, legendfontsize=16, legend=:topright, foreground_color_legend=nothing)

# Plot ACW-0 results
p2 = histogram(acw0_1, alpha=0.5, label="timescale 1 = $(timescale_1)",
    xtickfontsize=16, ytickfontsize=16, palette=colors, color=colors[5])
histogram!(p2, acw0_2, alpha=0.5, label="timescale 2 = $(timescale_2)", color=colors[4])

vline!(p2, [median(acw0_1), median(acw0_2)], linewidth=3, color=:black, label="")
title!(p2, "ACW-0\n", titlefontsize=24)
annotate!(p2, 2, 175, 
    (@sprintf("Proportion of \"wrong\" timescale \nestimates: %.2f%% \n", bad_acw0_timescale)), textfont=font(24), :left)


plot!(p2, legendfontsize=16, legend=:topright, foreground_color_legend=nothing)
plot(p1, p2, size=(1600, 800))
savefig("docs/src/practice/assets/practice_2_1.svg")

# Practice 2 Figure 2

timescale_1 = 1.0
timescale_2 = 3.0
sd = 1.0 
dt = 2.0 # Time interval between two time points
duration = 300.0 # 5 minutes of data
num_trials = 1000 # Number of trials

data_1 = generate_ou_process(timescale_1, sd, dt, duration, num_trials)
data_2 = generate_ou_process(timescale_2, sd, dt, duration, num_trials)

fs = 1 / dt # sampling rate
acwresults_1 = acw(data_1, fs, acwtypes=[:acw50, :acw0]) 
acwresults_2 = acw(data_2, fs, acwtypes=[:acw50, :acw0])
acw50_1 = acwresults_1.acw_results[1]
acw0_1 = acwresults_1.acw_results[2]
acw50_2 = acwresults_2.acw_results[1]
acw0_2 = acwresults_2.acw_results[2]

bad_acw50_timescale = mean(acw50_2 .<= acw50_1) * 100
bad_acw0_timescale = mean(acw0_2 .<= acw0_1) * 100

# Plot histograms
p1 = histogram(acw50_1, alpha=0.5, label="timescale 1 = $(timescale_1)")
histogram!(p1, acw50_2, alpha=0.5, label="timescale 2 = $(timescale_2)")
vline!(p1, [median(acw50_1), median(acw50_2)], linewidth=3, color=:black, label="") 
title!(p1, "ACW-50\n")
annotate!(p1, 3.0, 600, 
    (@sprintf("Proportion of \"wrong\" timescale \nestimates: %.2f%% \n", bad_acw50_timescale)), textfont=font(24), :left)
# ACW-0
p2 = histogram(acw0_1, alpha=0.5, label="timescale 1 = $(timescale_1)")
histogram!(p2, acw0_2, alpha=0.5, label="timescale 2 = $(timescale_2)")

vline!(p2, [median(acw0_1), median(acw0_2)], linewidth=3, color=:black, label="")
title!(p2, "ACW-0\n")
annotate!(p2, 40, 200, 
    (@sprintf("Proportion of \"wrong\" timescale \nestimates: %.2f%% \n", bad_acw0_timescale)), textfont=font(24), :left)
plot(p1, p2, size=(1600, 800))
savefig("docs/src/practice/assets/practice_2_2.svg")

# Practice 2 Figure 3

num_trials = 20 # Number of trials
data_1 = generate_ou_process(timescale_1, sd, dt, duration, num_trials)
data_2 = generate_ou_process(timescale_2, sd, dt, duration, num_trials)
acwresults_1 = acw(data_1, fs, acwtypes=[:acw50, :acw0]) 
acwresults_2 = acw(data_2, fs, acwtypes=[:acw50, :acw0])
p = acwplot(acwresults_1)
vline!(p, [acwresults_1.lags], linewidth=3, color=:black, label="")
savefig("docs/src/practice/assets/practice_2_3.svg")

# Practice 2 Figure 4

using Statistics, IntrinsicTimescales, Plots, Random
Random.seed!(123)
timescale = 3.0
sd = 1.0 # sd of data we'll simulate
dt = 0.001 # Time interval between two time points
fs = 1 / dt
duration = 10.0 # 10 seconds of data
num_trials = 1
acfs = []
acw50s = []
acw0s = []
n_experiments = 1000
for _ in 1:n_experiments
    data = generate_ou_process(timescale, sd, dt, duration, num_trials)
    acf = comp_ac_fft(data[:])
    push!(acfs, acf)
    current_mean_acf = mean(acfs)
    lags = (0:(length(current_mean_acf)-1)) * dt
    current_acw50 = acw50(lags, current_mean_acf)
    current_acw0 = acw0(lags, current_mean_acf)
    push!(acw50s, current_acw50)
    push!(acw0s, current_acw0)
end
p1 = plot(acw50s, label="ACW-50", xlabel="Iterations", ylabel="ACW", palette=colors, color=colors[5])
p2 = plot(acw0s, label="ACW-0", xlabel="Iterations", ylabel="ACW", palette=colors, color=colors[4])
plot(p1, p2, size=(800, 400))
savefig("docs/src/practice/assets/practice_2_4.svg")

# Practice 2 Figure 5

using HypothesisTests
Random.seed!(123)
n_subjects = 100
n_trials = 20
num_trials = 20
timescale_1 = 1.0
timescale_2 = 3.0
sd = 1.0
dt = 0.001
fs = 1 / dt
duration = 10.0

acw50_1 = Float64[] # HypothesisTests doesn't accept Vector{Any} type, requires Vector{<:Real} type
acw50_2 = Float64[]
acw0_1 = Float64[]
acw0_2 = Float64[]
for _ in 1:n_subjects
    data_1 = generate_ou_process(timescale_1, sd, dt, duration, num_trials)
    data_2 = generate_ou_process(timescale_2, sd, dt, duration, num_trials)
    acwresults_1 = acw(data_1, fs, acwtypes=[:acw50, :acw0], average_over_trials=true)
    acwresults_2 = acw(data_2, fs, acwtypes=[:acw50, :acw0], average_over_trials=true)
    current_acw50_1 = acwresults_1.acw_results[1]
    current_acw50_2 = acwresults_2.acw_results[1]
    current_acw0_1 = acwresults_1.acw_results[2]
    current_acw0_2 = acwresults_2.acw_results[2]
    push!(acw50_1, current_acw50_1)
    push!(acw50_2, current_acw50_2)
    push!(acw0_1, current_acw0_1)
    push!(acw0_2, current_acw0_2)
end

bad_acw50_timescale = mean(acw50_2 .<= acw50_1) * 100
bad_acw0_timescale = mean(acw0_2 .<= acw0_1) * 100

# Plot histograms
p1 = histogram(acw50_1, alpha=0.5, label="timescale 1 = $(timescale_1)", palette=colors, color=colors[5])
histogram!(p1, acw50_2, alpha=0.5, label="timescale 2 = $(timescale_2)", palette=colors, color=colors[4])
# Plot the median since distributions are not normal
vline!(p1, [median(acw50_1), median(acw50_2)], linewidth=3, color=:black, label="") 
title!(p1, "ACW-50\n")
annotate!(p1, 0.5, 25, 
    (@sprintf("Proportion of \"wrong\" timescale \nestimates: %.2f%% \n", bad_acw50_timescale)), textfont=font(24), :left)
# ACW-0
p2 = histogram(acw0_1, alpha=0.5, label="timescale 1 = $(timescale_1)")
histogram!(p2, acw0_2, alpha=0.5, label="timescale 2 = $(timescale_2)")

vline!(p2, [median(acw0_1), median(acw0_2)], linewidth=3, color=:black, label="")
title!(p2, "ACW-0\n")
annotate!(p2, 2, 30, 
    (@sprintf("Proportion of \"wrong\" timescale \nestimates: %.2f%% \n", bad_acw0_timescale)), textfont=font(12), :left)
plot(p1, p2, size=(1600, 800))

println(UnequalVarianceTTest(acw50_1, acw50_2))
println(UnequalVarianceTTest(acw0_1, acw0_2))
savefig("docs/src/practice/assets/practice_2_5.svg")
