using Test
using Statistics
using Distributions
using BayesianINT
using BayesianINT.OneTimescale
using BayesianINT.Models

@testset "OneTimescaleModel Tests" begin
    # Setup test data and model
    test_data = randn(10, 5000)  # 10 trials, 5000 timepoints
    test_priors = [
        Uniform(0.1, 10.0),  # tau prior
    ]
    n_lags = 3000

    model = OneTimescaleModel(
        test_data,           # data
        test_priors,         # prior
        zeros(n_lags),       # placeholder for data_sum_stats
        0.1,                 # epsilon
        0.01,               # deltaT
        0.01,               # binSize
        100.0,              # T
        10,                 # numTrials
        0.0,                # data_mean
        1.0,                # data_var
        n_lags              # n_lags
    )

    @testset "Model Construction" begin
        @test model isa OneTimescaleModel
        @test model isa AbstractTimescaleModel
        @test size(model.data) == (10, 5000)
        @test length(model.prior) == 1
        @test model.prior[1] isa Uniform
    end

    @testset "generate_data" begin
        theta = 1.0  # test parameter (tau)
        simulated_data = Models.generate_data(model, theta)
        
        @test size(simulated_data) == (model.numTrials, Int(model.T/model.deltaT))
        @test !any(isnan, simulated_data)
        @test !any(isinf, simulated_data)
        
        # Test statistical properties
        @test abs(mean(simulated_data)) < 0.1  # Should be close to 0
        @test abs(std(simulated_data) - model.data_var) < 0.1  # Should be close to data_var
    end

    @testset "summary_stats" begin
        # Test with both real and simulated data
        theta = 1.0
        simulated_data = Models.generate_data(model, theta)
        
        stats_real = Models.summary_stats(model, test_data)
        stats_sim = Models.summary_stats(model, simulated_data)
        
        @test length(stats_real) == model.n_lags
        @test length(stats_sim) == model.n_lags
        @test !any(isnan, stats_real)
        @test !any(isnan, stats_sim)
        @test all(stats_real .<= 1.0)  # Autocorrelation should be bounded by 1
        @test all(stats_sim .<= 1.0)
    end

    @testset "distance_function" begin
        # Generate two sets of summary statistics
        theta1 = 1.0
        theta2 = 2.0
        
        data1 = Models.generate_data(model, theta1)
        data2 = Models.generate_data(model, theta2)
        
        stats1 = Models.summary_stats(model, data1)
        stats2 = Models.summary_stats(model, data2)
        
        distance = Models.distance_function(model, stats1, stats2)
        
        @test distance isa Float64
        @test distance >= 0.0  # Distance should be non-negative
        @test Models.distance_function(model, stats1, stats1) ≈ 0.0 atol=1e-10
        
        # Test that different timescales lead to different distances
        @test distance > Models.distance_function(model, stats1, stats1)
    end

    @testset "Model Behavior" begin
        # Test that different taus produce different autocorrelations
        theta1 = 0.5
        theta2 = 2.0
        
        data1 = Models.generate_data(model, theta1)
        data2 = Models.generate_data(model, theta2)
        
        ac1 = Models.summary_stats(model, data1)
        ac2 = Models.summary_stats(model, data2)
        
        # The process with larger tau should have slower decay
        # Test this by comparing autocorrelations at some lag
        test_lag = 100
        @test ac2[test_lag] > ac1[test_lag]
    end
end 