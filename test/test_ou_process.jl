# test/test_ou_process.jl
using Test
using Statistics
using Distributions
using BayesianINT

@testset "OU Process Generation" begin
    @testset "Basic properties" begin
        tau = 0.5
        D = 1.0/tau
        deltaT = 0.001
        T = 1000.0
        num_trials = 100
        
        ou = generate_ou_process(tau, D, deltaT, T, num_trials)

        theoretical_var = D^2 * 0.5tau
        
        # Test dimensions
        @test size(ou) == (num_trials, Int(T/deltaT))
        
        # Test mean and variance
        @test mean(mean(ou, dims=2)) < 0.1  # Should be close to 0
        @test mean(abs.(var(ou, dims=2) .- theoretical_var)) < 0.1  # Should be close to 1
        
        # Test autocorrelation
        ac = mean([cor(@view(ou[i,1:end-1]), @view(ou[i,2:end])) for i in 1:num_trials])
        theoretical_ac = exp(-deltaT/tau)
        @test mean(abs.(ac .- theoretical_ac)) < 0.1

        # Whole ACF f'n
        ac = comp_ac_fft(ou[1:10, :])
        lags = range(deltaT, T-deltaT; step=deltaT)
        theoretical_ac = exp.(-lags/tau)
        @test mean(abs.(ac .- theoretical_ac)) < 0.05
    end
    
    @testset "Edge cases" begin
        # Test very small tau
        @test_nowarn generate_ou_process(0.1, 10.0, 0.01, 1.0, 10)
        
        # Test very large tau
        @test_nowarn generate_ou_process(1000.0, 0.001, 1.0, 100.0, 10)
        
        # Test single trial
        ou_single = generate_ou_process(20.0, 0.05, 1.0, 100.0, 1)
        @test size(ou_single, 1) == 1
    end
end

using Test
using Distributions

@testset "OneTimescaleModel Tests" begin
    # Setup test data and model
    test_data = randn(10, 100)  # 10 trials, 100 timepoints
    test_priors = [
        Uniform(0.1, 10.0),  # tau prior
        Uniform(0.1, 10.0)   # D prior
    ]
    
    model = OneTimescaleModel(
        test_data,           # data
        test_priors,         # prior
        zeros(10),          # placeholder for data_sum_stats
        0.1,                # epsilon
        0.01,              # deltaT
        0.01,              # binSize
        1.0,               # T
        10,                # numTrials
        0.0,               # data_mean
        1.0                # data_var
    )

    @testset "generate_data" begin
        theta = [1.0, 0.5]  # test parameters (tau, D)
        simulated_data = Models.generate_data(model, theta)
        
        @test size(simulated_data) == (model.numTrials, Int(model.T/model.deltaT))
        @test !any(isnan, simulated_data)
        @test !any(isinf, simulated_data)
    end

    @testset "summary_stats" begin
        # Test with both real and simulated data
        theta = [1.0, 0.5]
        simulated_data = Models.generate_data(model, theta)
        
        stats_real = Models.summary_stats(model, test_data)
        stats_sim = Models.summary_stats(model, simulated_data)
        
        @test length(stats_real) > 0
        @test length(stats_sim) > 0
        @test !any(isnan, stats_real)
        @test !any(isnan, stats_sim)
    end

    @testset "distance_function" begin
        # Generate two sets of summary statistics
        theta1 = [1.0, 0.5]
        theta2 = [2.0, 1.0]
        
        data1 = Models.generate_data(model, theta1)
        data2 = Models.generate_data(model, theta2)
        
        stats1 = Models.summary_stats(model, data1)
        stats2 = Models.summary_stats(model, data2)
        
        distance = Models.distance_function(model, stats1, stats2)
        
        @test distance isa Float64
        @test distance >= 0.0  # Distance should be non-negative
        @test Models.distance_function(model, stats1, stats1) ≈ 0.0 atol=1e-10
    end

    @testset "generate_ou_process" begin
        tau = 1.0
        D = 3
        deltaT = 0.001
        T = 10.0
        num_trials = 10
        
        ou_data = generate_ou_process(tau, D, deltaT, T, num_trials)
        
        @test size(ou_data) == (num_trials, Int(T/deltaT))
        @test !any(isnan, ou_data)
        @test !any(isinf, ou_data)
        
        # Test stationarity (mean should be approximately 0)
        @test mean(mean(ou_data, dims=2)) ≈ 0.0 atol=0.1
        
        # Test that variance is roughly as expected for OU process
        theoretical_var = D^2 * 0.5tau
        @test mean(abs.(var(ou_data, dims=2) .- theoretical_var)) < 0.11  # Should be close to 1
    end
end