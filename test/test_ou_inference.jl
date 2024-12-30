using Test
using Distributions
using BayesianINT
using Statistics

@testset "OU Parameter Inference" begin
    # Generate synthetic data with known parameters
    true_tau = 15.0
    true_D = 3.0
    dt = 0.01
    T = 100.0
    num_trials = 10
    n_lags = 3000
    
    # Generate synthetic data
    data = generate_ou_process(true_tau, true_D, dt, T, num_trials)
    
    # Set up priors
    priors = [
        Uniform(1.0, 30.0),  # tau prior
    ]
    data_acf = comp_ac_fft(data; n_lags=n_lags)
    
    # Create model
    model = OneTimescaleModel(
        data,               # data
        priors,            # prior
        data_acf,          # data_sum_stats
        1.0,               # epsilon
        dt,            # dt
        dt,            # binSize
        T,                 # T
        num_trials,        # numTrials
        mean(data),        # data_mean
        std(data),         # data_var
        n_lags             # n_lags
    )
    
    # Run PMC-ABC
    results = pmc_abc(
        model;
        epsilon_0=0.01,
        min_samples=100,
        steps=3,
        minAccRate=0.001,
        max_iter=1000
    )
    
    # Get final posterior samples
    final_samples = results[end].theta_accepted
    
    # Calculate posterior means
    posterior_tau = mean(final_samples[1,:])
    tau_std = std(final_samples[1,:])
    
    # Test if estimates are within reasonable range
    @test abs(posterior_tau - true_tau) < 5.0  # Within 10 units of true tau

end
