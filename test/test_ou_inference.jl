using Test
using Distributions
using BayesianINT
using Statistics
using BayesianINT.OrnsteinUhlenbeck

@testset "OU Parameter Inference" begin
    # Figure 3a of Zeraati et al. paper
    # Generate synthetic data with known parameters
    true_tau = 20.0 / 1000.0
    true_D = 3.0
    dt = 1 / 1000
    T = 1.0
    num_trials = 500
    n_lags = 50

    # Generate synthetic data
    data = generate_ou_process(true_tau, true_D, dt, T, num_trials)

    # Set up priors
    priors = [
        Uniform(1.0 / 1000.0, 30.0 / 1000.0)  # tau prior
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
    results = bayesian_inference(model;
                      epsilon_0=0.1,
                      min_samples=100,
                      steps=10,
                      minAccRate=0.001,
                      max_iter=500)

    # Get final posterior samples
    final_samples = results[end].theta_accepted

    # Calculate posterior means
    posterior_tau = mean(final_samples[1, :])
    tau_std = std(final_samples[1, :])

    # Test if estimates are within reasonable range
    @test abs(posterior_tau - true_tau) < 1.0
end

@testset "OU with Oscillation Parameter Inference" begin
    # Generate synthetic data with known parameters
    true_tau = 100.0 / 1000.0
    true_freq = 10.0  # Hz
    true_coeff = 0.3  # oscillation coefficient
    dt = 1 / 1000
    T = 30.0
    num_trials = 10
    n_lags = 50
    epsilon_0 = 1.0

    # Calculate data mean and variance (needed for OneTimescaleAndOscModel)
    data = generate_ou_with_oscillation([true_tau, true_freq, true_coeff],
                                        dt, T, num_trials, 0.0, 1.0)
    data_mean = mean(data)
    data_var = std(data)

    # Set up priors
    priors = [
        Uniform(30.0 / 1000.0, 120.0 / 1000.0),  # tau prior
        Uniform(1.0, 60.0),                      # frequency prior
        Uniform(0.0, 1.0)                        # oscillation coefficient prior
    ]

    data_sum_stats = comp_psd(data, 1/dt)[1]

    # Create model
    model = OneTimescaleAndOscModel(data,              # data
                                    priors,           # prior
                                    data_sum_stats,   # data_sum_stats
                                    epsilon_0,        # epsilon
                                    dt,               # dt
                                    T,                # T
                                    num_trials,       # numTrials
                                    data_mean,        # data_mean
                                    data_var,         # data_var
                                    )

    # Run PMC-ABC
    results = Models.bayesian_inference(model;
                      epsilon_0=0.5,
                      min_samples=100,
                      steps=100,
                      minAccRate=0.001,
                      max_iter=500)

    # Get final posterior samples
    final_samples = results[end].theta_accepted

    # Calculate posterior means/MAPs
    posterior_tau = mean(final_samples[1, :])
    posterior_freq = mean(final_samples[2, :])
    posterior_coeff = mean(final_samples[3, :])

    sd_tau = std(final_samples[1, :])
    sd_freq = std(final_samples[2, :])
    sd_coeff = std(final_samples[3, :])

    # Test if estimates are within reasonable range
    @test abs(posterior_tau - true_tau) < 10.0 / 1000.0
    @test abs(posterior_freq - true_freq) < 3.0
    @test abs(posterior_coeff - true_coeff) < 0.2
end

using Plots
ou_final = generate_ou_with_oscillation([posterior_tau, posterior_freq, posterior_coeff], dt, T, num_trials, 0.0, 1.0)
ou_final_sum_stats, freq = comp_psd(ou_final, 1/dt)

plot(freq[2:end], data_sum_stats, scale=:ln, label="Data")
plot!(freq[2:end], ou_final_sum_stats, scale=:ln, label="Model")
