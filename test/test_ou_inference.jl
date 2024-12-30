using Test
using Distributions
using BayesianINT
using Statistics

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
        Uniform(1.0 / 1000.0, 30.0 / 1000.0),  # tau prior
    ]
    data_acf = comp_ac_fft(data; n_lags=n_lags)
    
    # Create model
    model = OneTimescaleModel(
        data,              # data
        priors,            # prior
        data_acf,          # data_sum_stats
        1.0,               # epsilon
        dt,                # dt
        T,                 # T
        num_trials,        # numTrials
        std(data),         # data_var
        n_lags             # n_lags
    )
    
    # Run PMC-ABC
    results = pmc_abc(
        model;
        epsilon_0=0.1,
        min_samples=100,
        steps=10,
        minAccRate=0.001,
        max_iter=500
    )
    
    # Get final posterior samples
    final_samples = results[end].theta_accepted
    
    # Calculate posterior means
    posterior_tau = find_MAP(final_samples[1,:], 20000)
    tau_std = std(final_samples[1,:])
    
    # Test if estimates are within reasonable range
    @test abs(posterior_tau - true_tau) < 1.0

end
