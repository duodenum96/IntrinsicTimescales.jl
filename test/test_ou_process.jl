# test/test_ou_process.jl
using Test
using Statistics
using Distributions
using BayesianINT

@testset "OU Process Generation" begin
    @testset "Basic properties" begin
        for backend in ["vanilla", "sciml"]
            @testset "$backend backend" begin
                tau = 0.5
                D = 2.0
                deltaT = 0.001
                T = 10.0
                num_trials = 100
                n_lags = 3000
                
                ou = generate_ou_process(tau, D, deltaT, T, num_trials, backend=backend)

                # Test dimensions
                @test size(ou) == (num_trials, Int(T/deltaT))
                
                # Test mean and variance
                @test mean(mean(ou, dims=2)) < 0.1  # Should be close to 0
                @test mean(abs.(std(ou, dims=2) .- D)) < 0.1  # Should be close to D
                
                # Test autocorrelation
                ac = mean([cor(@view(ou[i,1:end-1]), @view(ou[i,2:end])) for i in 1:num_trials])
                theoretical_ac = exp(-deltaT/tau)
                @test mean(abs.(ac .- theoretical_ac)) < 0.1

                # Whole ACF f'n
                ac = comp_ac_fft(ou[1:20, :]; n_lags=n_lags)
                lags = range(0, T-deltaT; step=deltaT)
                theoretical_ac = exp.(-lags/tau)[1:n_lags]
                @test mean(sqrt.(abs2.(ac .- theoretical_ac))) < 0.11 # What's a good number here?
            end
        end
    end
    
    @testset "Edge cases" begin
        for backend in ["vanilla", "sciml"]
            @testset "$backend backend" begin
                # Test very small tau
                @test_nowarn generate_ou_process(0.1, 10.0, 0.01, 1.0, 10, backend=backend)
                
                # Test very large tau
                @test_nowarn generate_ou_process(1000.0, 0.001, 1.0, 100.0, 10, backend=backend)
                
                # Test single trial
                ou_single = generate_ou_process(20.0, 0.05, 1.0, 100.0, 1, backend=backend)
                @test size(ou_single, 1) == 1
            end
        end
    end

    @testset "Backend comparison" begin
        tau = 0.5
        D = 2.0
        deltaT = 0.001
        T = 10.0  # Reduced from 1000.0 for faster testing
        num_trials = 100
        
        # Generate data with both backends
        ou_vanilla = generate_ou_process(tau, D, deltaT, T, num_trials, backend="vanilla")
        ou_sciml = generate_ou_process(tau, D, deltaT, T, num_trials, backend="sciml")

        # Test dimensions match
        @test size(ou_vanilla) == size(ou_sciml)
        
        # Test statistical properties are similar between implementations
        @test abs(mean(ou_vanilla) - mean(ou_sciml)) < 0.1
        @test abs(std(ou_vanilla) - std(ou_sciml)) < 0.1
        
        # Test autocorrelations are similar
        ac_vanilla = mean([cor(@view(ou_vanilla[i,1:end-1]), @view(ou_vanilla[i,2:end])) for i in 1:num_trials])
        ac_sciml = mean([cor(@view(ou_sciml[i,1:end-1]), @view(ou_sciml[i,2:end])) for i in 1:num_trials])
        @test abs(ac_vanilla - ac_sciml) < 0.1
    end

    # Test invalid backend
    @test_throws ErrorException generate_ou_process(0.5, 2.0, 0.001, 10.0, 100, backend="invalid")

    # Test parallel execution for SciML
    @testset "Parallel execution" begin
        tau = 0.5
        D = 2.0
        deltaT = 0.001
        T = 10.0
        num_trials = 100
        
        ou_parallel = generate_ou_process(tau, D, deltaT, T, num_trials, backend="sciml", parallel=true)
        ou_serial = generate_ou_process(tau, D, deltaT, T, num_trials, backend="sciml", parallel=false)
        
        @test size(ou_parallel) == size(ou_serial)
        @test abs(mean(ou_parallel) - mean(ou_serial)) < 0.1
    end
end

using Test
using Distributions

@testset "OneTimescaleModel Tests" begin
    # Setup test data and model
    test_data = randn(10, 5000)  # 10 trials, 100 timepoints
    test_priors = [
        Uniform(0.1, 10.0),  # tau prior
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
        theta = 1.0  # test parameters (tau, D)
        simulated_data = Models.generate_data(model, theta)
        
        @test size(simulated_data) == (model.numTrials, Int(model.T/model.deltaT))
        @test !any(isnan, simulated_data)
        @test !any(isinf, simulated_data)
    end

    @testset "summary_stats" begin
        # Test with both real and simulated data
        theta = 1.0
        simulated_data = Models.generate_data(model, theta)
        
        stats_real = Models.summary_stats(model, test_data; n_lags=20)
        stats_sim = Models.summary_stats(model, simulated_data; n_lags=20)
        
        @test length(stats_real) > 0
        @test length(stats_sim) > 0
        @test !any(isnan, stats_real)
        @test !any(isnan, stats_sim)
    end

    @testset "distance_function" begin
        # Generate two sets of summary statistics
        theta1 = 1.0
        theta2 = 2.0
        
        data1 = Models.generate_data(model, theta1)
        data2 = Models.generate_data(model, theta2)
        
        stats1 = Models.summary_stats(model, data1; n_lags=20)
        stats2 = Models.summary_stats(model, data2; n_lags=20)
        
        distance = Models.distance_function(model, stats1, stats2; n_lags=20)
        
        @test distance isa Float64
        @test distance >= 0.0  # Distance should be non-negative
        @test Models.distance_function(model, stats1, stats1; n_lags=20) ≈ 0.0 atol=1e-10
    end

    @testset "generate_ou_process" begin
        tau = 1.0
        D = 3.0
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
        @test mean(abs.(std(ou_data, dims=2) .- D)) < 0.1  # Should be close to 1
    end
end