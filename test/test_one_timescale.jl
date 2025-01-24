using Test
using Statistics
using Distributions
using BayesianINT
using BayesianINT.OneTimescale
using BayesianINT.Models

@testset "OneTimescale Model Tests" begin
    # Setup test data and parameters
    dt = 0.01
    T = 100.0
    num_trials = 10
    n_lags = 3000
    ntime = Int(T / dt)
    time = 0:dt:T
    test_data = randn(num_trials, ntime)  # 10 trials, 10000 timepoints
    data_mean = mean(test_data)
    data_sd = std(test_data)
    
    @testset "Model Construction - ACF and ABC" begin
        model = one_timescale_model(
            test_data,
            time,
            :abc;
            summary_method=:acf,
            prior=[Uniform(0.1, 10.0)],  # tau prior
            n_lags=n_lags,
            distance_method=:linear
        )

        @test model isa OneTimescaleModel
        @test model isa AbstractTimescaleModel
        @test size(model.data) == (num_trials, ntime)
        @test model.prior[1] isa Uniform
        @test model.fit_method == :abc
        @test model.summary_method == :acf
        @test model.distance_method == :linear
    end

    @testset "Model Construction - PSD and ABC" begin
        model = one_timescale_model(
            test_data,
            time,
            :abc;
            summary_method=:psd,
            prior=[Uniform(0.1, 10.0)],
            freqlims=(0.5, 100.0),
            distance_method=:logarithmic
        )

        @test model.summary_method == :psd
        @test model.freqlims == (0.5, 100.0)
        @test model.distance_method == :logarithmic
    end

    @testset "Model Construction - ACW" begin
        acw_results = one_timescale_model(
            test_data,
            time,
            :acw;
            acwtypes=[:acw0, :acw50, :acweuler, :tau, :knee]
        )

        @test length(acw_results) == 5
        @test all(x -> x isa AbstractArray{<:Real}, acw_results)
        @test all(all.(map(x -> x .> 0, acw_results)))  # All timescales should be positive
    end

    @testset "Informed Prior" begin
        model = one_timescale_model(
            test_data,
            time,
            :abc;
            summary_method=:acf,
            prior="informed_prior",
            n_lags=100
        )
        
        @test model.prior[1] isa Normal
    end

    @testset "generate_data" begin
        model = one_timescale_model(
            test_data,
            time,
            :abc;
            summary_method=:acf,
            n_lags=100
        )
        
        theta = [1.0]  # test parameter (tau)
        simulated_data = Models.generate_data(model, theta)
        
        @test size(simulated_data) == (model.numTrials, Int(model.T/model.dt))
        @test !any(isnan, simulated_data)
        @test !any(isinf, simulated_data)
        
        # Test statistical properties
        @test abs(std(simulated_data) - model.data_sd) < 0.1
        @test abs(mean(simulated_data) - model.data_mean) < 0.1
    end

    @testset "summary_stats and distance_function" begin
        model = one_timescale_model(
            test_data,
            time,
            :abc;
            summary_method=:acf,
            distance_method=:linear,
            n_lags=100
        )
        
        theta1 = [1.0]
        theta2 = [2.0]
        
        data1 = Models.generate_data(model, theta1)
        data2 = Models.generate_data(model, theta2)
        
        stats1 = Models.summary_stats(model, data1)
        stats2 = Models.summary_stats(model, data2)
        
        # Test summary stats properties
        @test length(stats1) == model.n_lags
        @test !any(isnan, stats1)
        @test stats1[1] ≈ 1.0 atol=0.1  # First lag should be close to 1
        @test all(abs.(stats1) .<= 1.0)  # All autocorrelations should be ≤ 1
        
        # Test distance function
        distance = Models.distance_function(model, stats1, stats2)
        @test distance isa Float64
        @test distance >= 0.0
        @test Models.distance_function(model, stats1, stats1) ≈ 0.0 atol=1e-10
        @test distance > Models.distance_function(model, stats1, stats1)
    end

    @testset "Combined Distance" begin
        model = one_timescale_model(
            test_data,
            time,
            :abc;
            summary_method=:acf,
            distance_method=:linear,
            distance_combined=true,
            weights=[0.7, 0.3],
            n_lags=100
        )

        @test model.distance_combined == true
        @test model.weights == [0.7, 0.3]
        @test !isnothing(model.data_tau)
    end
end
