using Test
using Distributions
using BayesianINT
using Statistics

@testset "OU Parameter Inference" begin
    # Generate synthetic data with known parameters
    true_tau = 15.0
    true_D = 3.0
    deltaT = 0.01
    T = 100.0
    num_trials = 10
    
    # Generate synthetic data
    data = generate_ou_process(true_tau, true_D, deltaT, T, num_trials)
    
    # Set up priors
    priors = [
        Uniform(1.0, 30.0),  # tau prior
    ]
    
    # Create model
    model = OneTimescaleModel(
        data,               # data
        priors,            # prior
        comp_ac_fft(data), # data_sum_stats
        1.0,               # epsilon
        deltaT,            # deltaT
        deltaT,            # binSize
        T,                 # T
        num_trials,        # numTrials
        mean(data),        # data_mean
        std(data)          # data_var
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

    # Test if posterior variance decreased
    @test var(final_samples[1,:]) < var(results[1].theta_accepted[1,:])
end
