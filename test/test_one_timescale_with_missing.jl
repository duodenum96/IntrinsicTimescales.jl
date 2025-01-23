using Test
using Statistics
using Distributions
using BayesianINT
using BayesianINT.OrnsteinUhlenbeck
using BayesianINT.OneTimescaleWithMissing
using BayesianINT.Models

@testset "OneTimescaleWithMissing Model Tests" begin
    # Setup test data and model parameters
    dt = 0.01
    T = 100.0
    num_trials = 10
    n_lags = 3000
    ntime = Int(T / dt)
    
    # Generate test data with known parameters
    true_tau = 1.0
    test_data = generate_ou_process(true_tau, 1.0, dt, T, num_trials)
    
    # Add missing values (10% of data)
    missing_mask = rand(size(test_data)...) .< 0.1
    test_data[missing_mask] .= NaN
    
    # Compute data statistics
    data_var = mean([std(filter(!isnan, test_data[i, :])) for i in 1:num_trials])
    
    # Compute autocorrelation with missing data
    data_sum_stats = comp_ac_time_missing(test_data, n_lags)

    data_sum_stats_mean = mean(data_sum_stats, dims=1)[:]
    
    # Define test priors
    test_prior = [Uniform(0.1, 10.0)]  # tau prior

    model = OneTimescaleWithMissingModel(
        test_data,           # data
        test_prior,         # prior
        data_sum_stats_mean,     # data_sum_stats
        0.1,                # epsilon
        dt,                 # dt
        T,                  # T
        num_trials,         # numTrials
        data_var,           # data_var
        n_lags              # n_lags
    )

    @testset "Model Construction" begin
        @test model isa OneTimescaleWithMissingModel
        @test model isa AbstractTimescaleModel
        @test size(model.data) == (num_trials, ntime)
        @test length(model.prior) == 1  # single tau parameter
        @test model.prior[1] isa Uniform
        @test size(model.missing_mask) == size(test_data)
        @test model.missing_mask == missing_mask
    end

    @testset "generate_data" begin
        theta = [1.0]  # test parameter (tau)
        simulated_data = Models.generate_data(model, theta)
        
        @test size(simulated_data) == (model.numTrials, Int(model.T/model.dt))
        @test all(isnan.(simulated_data[model.missing_mask]))
        @test !all(isnan.(simulated_data[.!model.missing_mask]))
        
        # Test statistical properties of non-missing data
        valid_data = filter(!isnan, simulated_data)
        @test abs(std(valid_data) - sqrt(model.data_var)) < 0.2
        @test abs(mean(valid_data)) < 0.2  # Should be close to zero
    end

    @testset "summary_stats" begin
        theta = [1.0]
        simulated_data = Models.generate_data(model, theta)
        
        stats = Models.summary_stats(model, simulated_data)
        stats_mean = mean(stats, dims=1)[:]

        @test length(stats_mean) == model.n_lags
        @test !any(isnan, stats_mean)
        @test stats_mean[1] ≈ 1.0 atol=0.1  # First lag should be close to 1
        @test all((abs.(stats_mean) .<= 1.0) .| (abs.(stats_mean) .≈ 1.0))  # All autocorrelations should be ≤ 1
        @test issorted(stats_mean[1:200], rev=true) 
    end

    @testset "distance_function" begin
        theta1 = [1.0]
        theta2 = [2.0]
        
        data1 = Models.generate_data(model, theta1)
        data2 = Models.generate_data(model, theta2)
        
        stats1 = Models.summary_stats(model, data1)
        stats2 = Models.summary_stats(model, data2)
        
        distance = Models.distance_function(model, stats1, stats2)
        
        @test distance isa Float64
        @test distance >= 0.0
        @test Models.distance_function(model, stats1, stats1) ≈ 0.0 atol=1e-10
        
        # Test that different parameters lead to different distances
        @test distance > Models.distance_function(model, stats1, stats1)
    end

    @testset "Model Behavior" begin
        # Test effect of different timescales
        theta1 = [0.5]  # faster timescale
        theta2 = [2.0]  # slower timescale
        
        data1 = Models.generate_data(model, theta1)
        data2 = Models.generate_data(model, theta2)
        
        ac1 = Models.summary_stats(model, data1)
        ac2 = Models.summary_stats(model, data2)
        
        # Test that slower timescale has higher autocorrelation at longer lags
        lag_idx = 100  # Compare at lag 100
        @test ac2[lag_idx] > ac1[lag_idx]
    end
end 