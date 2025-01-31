using Plots

colors = palette(:Catppuccin_frappe)[[4, 5, 7, 9, 3, 10, 13]]

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
using INT # import INT package
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