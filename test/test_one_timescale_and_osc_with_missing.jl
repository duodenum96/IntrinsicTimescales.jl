using Test
using Statistics
using Distributions
using BayesianINT
using BayesianINT.OrnsteinUhlenbeck
using BayesianINT.OneTimescaleAndOscWithMissing
using BayesianINT.Models

@testset "OneTimescaleAndOscWithMissing Model Tests" begin
    # Setup test data and model parameters
    dt = 0.01
    T = 100.0
    num_trials = 10
    ntime = Int(T / dt)
    times = collect(0:dt:T-dt)
    
    # Generate test data with known parameters
    true_tau = 1.0
    true_freq = 0.1  # Hz
    true_coeff = 0.5
    test_data = generate_ou_with_oscillation(
        [true_tau, true_freq, true_coeff],
        dt, T, num_trials, 0.0, 1.0
    )
    
    # Add missing values (10% of data)
    missing_mask = rand(size(test_data)...) .< 0.1
    test_data[missing_mask] .= NaN
    
    # Compute data statistics
    data_mean = mean(filter(!isnan, test_data))
    data_var = var(filter(!isnan, test_data))
    
    # Compute PSD using Lomb-Scargle
    data_sum_stats = comp_psd_lombscargle(times, test_data, missing_mask, dt)
    
    data_sum_stats = mean(data_sum_stats[1], dims=1)[:], data_sum_stats[2]
    # Define test priors
    test_prior = [
        Uniform(0.1, 10.0),  # tau prior
        Uniform(0.01, 1.0),  # frequency prior
        Uniform(0.0, 1.0)    # amplitude prior
    ]

    model = OneTimescaleAndOscWithMissingModel(
        test_data,           # data
        times,              # times
        test_prior,         # prior
        data_sum_stats,     # data_sum_stats
        0.1,                # epsilon
        dt,                 # dt
        T,                  # T
        num_trials,         # numTrials
        data_mean,          # data_mean
        data_var            # data_var
    )

    @testset "Model Construction" begin
        @test model isa OneTimescaleAndOscWithMissingModel
        @test model isa AbstractTimescaleModel
        @test size(model.data) == (num_trials, ntime)
        @test length(model.prior) == 3  # tau, frequency, amplitude
        @test all(p isa Uniform for p in model.prior)
        @test size(model.missing_mask) == size(test_data)
        @test model.missing_mask == missing_mask
    end

    @testset "Informed Prior Construction" begin
        informed_model = OneTimescaleAndOscWithMissingModel(
            test_data,
            times,
            "informed",
            data_sum_stats,
            0.1,
            dt,
            T,
            num_trials,
            data_mean,
            data_var
        )
        @test informed_model.prior[1] isa Normal  # tau prior
        @test informed_model.prior[2] isa Normal  # frequency prior
        @test informed_model.prior[3] isa Uniform # amplitude prior
    end

    @testset "generate_data" begin
        theta = [1.0, 0.1, 0.5]  # test parameters (tau, freq, amplitude)
        simulated_data = Models.generate_data(model, theta)
        
        @test size(simulated_data) == (model.numTrials, Int(model.T/model.dt))
        @test all(isnan.(simulated_data[model.missing_mask]))
        @test !all(isnan.(simulated_data[.!model.missing_mask]))
        
        # Test statistical properties of non-missing data
        valid_data = filter(!isnan, simulated_data)
        @test abs(mean(valid_data) - model.data_mean) < 0.2
        @test abs(std(valid_data) - sqrt(model.data_var)) < 0.2
    end

    @testset "summary_stats" begin
        theta = [1.0, 0.1, 0.5]
        simulated_data = Models.generate_data(model, theta)
        
        stats = Models.summary_stats(model, simulated_data)
        
        @test length(stats) == 2  # Should return (psd, freqs)
        @test !any(isnan, stats[1])  # PSD should not contain NaNs
        @test !any(isnan, stats[2])  # Frequencies should not contain NaNs
        @test size(stats[1], 2) == length(stats[2])  # PSD and freq vectors should match
        @test all(stats[2] .>= 0)  # Frequencies should be non-negative
    end

    @testset "distance_function" begin
        theta1 = [1.0, 0.1, 0.5]
        theta2 = [2.0, 0.2, 0.7]
        
        data1 = Models.generate_data(model, theta1)
        data2 = Models.generate_data(model, theta2)
        
        stats1 = Models.summary_stats(model, data1)
        stats2 = Models.summary_stats(model, data2)

        stats1_mean = mean(stats1[1], dims=1)[:], stats1[2]
        stats2_mean = mean(stats2[1], dims=1)[:], stats2[2]
        
        distance = Models.distance_function(model, stats1_mean, stats2_mean)
        
        @test distance isa Float64
        @test distance >= 0.0
        @test Models.distance_function(model, stats1_mean, stats1_mean) < distance
    end

    @testset "Model Behavior" begin
        # Test effect of different parameters
        theta_base = [1.0, 0.1, 0.5]
        theta_higher_freq = [1.0, 0.2, 0.5]
        
        data_base = Models.generate_data(model, theta_base)
        data_higher_freq = Models.generate_data(model, theta_higher_freq)
        
        stats_base = Models.summary_stats(model, data_base)
        stats_higher = Models.summary_stats(model, data_higher_freq)
        
        stats_base_mean = mean(stats_base[1], dims=1)[:], stats_base[2]
        stats_higher_mean = mean(stats_higher[1], dims=1)[:], stats_higher[2]
        
        # Find peak frequencies
        peak_freq_base = stats_base_mean[2][argmax(stats_base_mean[1])]
        peak_freq_higher = stats_higher_mean[2][argmax(stats_higher_mean[1])]
        
        # Test that higher frequency parameter leads to higher peak frequency
        @test peak_freq_higher > peak_freq_base
    end
end 