using Test
using Distributions
using BayesianINT
using Statistics
using BayesianINT
using BayesianINT.Models
using BayesianINT.OrnsteinUhlenbeck
using LinearAlgebra
BLAS.set_num_threads(16)
# using Plots

@testset "OU Parameter Inference" begin
    # Figure 3a of Zeraati et al. paper
    # Generate synthetic data with known parameters (units are in ms)
    true_tau = 20.0
    true_D = 3.0
    dt = 1.0
    T = 1000.0
    num_trials = 500
    n_lags = 50

    # Generate synthetic data
    data = generate_ou_process(true_tau, true_D, dt, T, num_trials)

    # Set up priors
    priors = [
        Uniform(1.0, 100.0)  # tau prior
    ]
    data_acf = comp_ac_fft(data; n_lags=n_lags)

    # Create model
    model = OneTimescaleModel(data,              # data
                              priors,            # prior
                              data_acf,          # data_sum_stats
                              1.0,               # epsilon
                              dt,                # dt
                              T,                 # T
                              num_trials,        # numTrials
                              std(data),         # data_var
                              n_lags)

    # Run PMC-ABC
    timex = @elapsed results = pmc_abc(model;
                      epsilon_0=0.01,
                      min_accepted=100,
                      steps=60,
                      minAccRate=0.01,
                      max_iter=10000,
                      target_epsilon=1e-4)

    println("Time taken: $timex seconds")
    # Get final posterior samples
    final_samples = results[end].theta_accepted

    # Calculate posterior means
    N_MAP = 10000
    posterior_MAP = find_MAP(final_samples, N_MAP)
    posterior_tau = posterior_MAP[1]
    tau_std = std(final_samples[:, 1])

    # Test if estimates are within reasonable range
    @test abs(posterior_tau - true_tau) < 3.0
    # Plot
    # histogram(final_samples[1, :])
    # vline!([true_tau])
end

@testset "OU with Oscillation Parameter Inference" begin
    # Generate synthetic data with known parameters
    true_tau = 100.0
    true_freq = 10.0 / 1000.0  # mHz
    true_coeff = 0.7  # oscillation coefficient
    dt = 1.0
    T = 1000.0
    num_trials = 100
    n_lags = 100
    epsilon_0 = 1.0

    # Calculate data mean and variance (needed for OneTimescaleAndOscModel)
    data = generate_ou_with_oscillation([true_tau, true_freq, true_coeff],
                                        dt, T, num_trials, 0.0, 1.0)
    data_mean = mean(data)
    data_var = std(data)

    data_psd, data_freq = comp_psd(data, 1/dt)
    # priors = informed_prior(model, data_freq)

    # Create model
    model = OneTimescaleAndOscModel(data,              # data
                                    "informed",       # prior
                                    [data_psd, data_freq],   # data_sum_stats
                                    epsilon_0,        # epsilon
                                    dt,               # dt
                                    T,                # T
                                    num_trials,       # numTrials
                                    data_mean,        # data_mean
                                    data_var)         # data_var

    # Run PMC-ABC
    timex = @elapsed results = pmc_abc(model;
                      epsilon_0=0.5,
                      min_accepted=100,
                      steps=10,
                      minAccRate=0.01,
                      max_iter=10000,
                      target_epsilon=1e-2)

    println("Time taken: $timex seconds")

    # Get final posterior samples
    final_samples = results[end].theta_accepted

    # Calculate posterior means/MAPs
    N_MAP = 10000
    posterior_MAP = find_MAP(final_samples, N_MAP)
    posterior_tau = posterior_MAP[1]
    posterior_freq = posterior_MAP[2]
    posterior_coeff = posterior_MAP[3]

    sd_tau = std(final_samples[:, 1])
    sd_freq = std(final_samples[:, 2])
    sd_coeff = std(final_samples[:, 3])

    # Test if estimates are within reasonable range
    @test abs(posterior_tau - true_tau) < 10.0
    @test abs(posterior_freq - true_freq) < 3.0 / 1000.0
    @test abs(posterior_coeff - true_coeff) < 0.2
    # Plot
    # ou_final = generate_ou_with_oscillation([posterior_tau, posterior_freq, posterior_coeff], dt, T, num_trials, 0.0, 1.0)
    # ou_final_sum_stats, freq = comp_psd(ou_final, 1/dt)
    # plot(freq, data_psd, scale=:log10, label="Data")
    # plot!(freq, ou_final_sum_stats, scale=:ln, label="Model")

    # histogram(final_samples[:, 1])
    # vline!([true_tau])
    # histogram(final_samples[:, 2])
    # vline!([true_freq])
    # histogram(final_samples[:, 3])
    # vline!([true_coeff])
    
end

@testset "OU with Missing Data" begin
    # Generate synthetic data with known parameters
    true_tau = 30.0
    true_D = 3.0
    dt = 1.0
    T = 1000.0
    num_trials = 100
    n_lags = 100

    # Generate synthetic data
    data = generate_ou_process(true_tau, true_D, dt, T, num_trials)
    missing_mask = rand(size(data)...) .< 0.1  # Makes 10% of entries true
    data[missing_mask] .= NaN

    # Set up priors
    priors = [
        Uniform(1.0, 100.0)  # tau prior
    ]
    data_acf = comp_ac_time_missing(data, n_lags)
    data_mean = mean(filter(!isnan, data))
    data_var = std(filter(!isnan, data))
    # Create model
    model = OneTimescaleWithMissingModel(data,              # data
                                        priors,            # prior
                                        data_acf,          # data_sum_stats
                                        1.0,               # epsilon
                                        dt,                # dt
                                        T,                 # T
                                        num_trials,        # numTrials
                                        data_var,        # data_var
                                        n_lags)
    
    # Run PMC-ABC
    timex = @elapsed results = pmc_abc(model;
                      epsilon_0=0.1,
                      min_accepted=100,
                      steps=60,
                      minAccRate=0.01,
                      max_iter=10000,
                      target_epsilon=1e-3)

    println("Time taken: $timex seconds")

    # Get final posterior samples
    final_samples = results[end].theta_accepted
    N_MAP = 10000
    # Calculate posterior means
    posterior_MAP = find_MAP(final_samples, N_MAP)
    posterior_tau = posterior_MAP[1]
    tau_std = std(final_samples[:, 1])

    # Test if estimates are within reasonable range
    @test abs(posterior_tau - true_tau) < 3.0
end
