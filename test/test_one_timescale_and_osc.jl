using Test
using Statistics
using Distributions
using BayesianINT
using BayesianINT.OrnsteinUhlenbeck
using BayesianINT.OneTimescaleAndOsc
using BayesianINT.Models

@testset "OneTimescaleAndOscModel Tests" begin
    # Setup test data and model
    test_priors = [
        Uniform(0.1, 10.0),  # tau prior
        Uniform(0.1, 5.0),   # amplitude prior
        Uniform(0.1, 2.0),   # frequency prior
    ]
    dt = 0.01
    T = 100.0
    num_trials = 10
    n_lags = 3000
    ntime = Int(T / dt)
    test_data = randn(num_trials, ntime)  # 10 trials, 5000 timepoints
    data_mean = mean(test_data)
    data_var = std(test_data)
    model = OneTimescaleAndOscModel(
        test_data,           # data
        test_priors,         # prior
        zeros(n_lags),       # placeholder for data_sum_stats
        0.1,                 # epsilon
        dt,                 # dt
        T,                  # T
        num_trials,         # numTrials
        data_mean,          # data_mean
        data_var,           # data_var
        n_lags              # n_lags
    )

    @testset "Model Construction" begin
        @test model isa OneTimescaleAndOscModel
        @test model isa AbstractTimescaleModel
        @test size(model.data) == (num_trials, Int(T / dt))
        @test length(model.prior) == 3  # tau, amplitude, frequency
        @test all(p isa Uniform for p in model.prior)
    end

    @testset "generate_data" begin
        theta = [1.0, 0.5, 1.0]  # test parameters (tau, amplitude, frequency)
        simulated_data = Models.generate_data(model, theta)
        
        @test size(simulated_data) == (model.numTrials, Int(model.T/model.dt))
        @test !any(isnan, simulated_data)
        @test !any(isinf, simulated_data)
        
        # Test statistical properties
        @test abs(mean(simulated_data) - model.data_mean) < 0.1
        @test abs(std(simulated_data) - sqrt(model.data_var)) < 0.1
    end

    @testset "summary_stats" begin
        theta = [1.0, 0.5, 1.0]
        simulated_data = Models.generate_data(model, theta)
        
        stats_real = Models.summary_stats(model, test_data)
        stats_sim = Models.summary_stats(model, simulated_data)
        
        @test !any(isnan, stats_real)
        @test !any(isnan, stats_sim)
    end

    @testset "distance_function" begin
        theta1 = [1.0, 0.5, 1.0]
        theta2 = [2.0, 0.3, 0.8]
        
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
        # Test effect of different parameters
        theta1 = [0.5, 0.5, 1.0]  # faster timescale
        theta2 = [2.0, 0.5, 1.0]  # slower timescale
        
        data1 = Models.generate_data(model, theta1)
        data2 = Models.generate_data(model, theta2)
        
        ac1 = Models.summary_stats(model, data1)
        ac2 = Models.summary_stats(model, data2)
        
        # TODO: Write an appropriate test for the PSD based calculations
        
    end
end