using Test
using Statistics
using Distributions
using INT
using INT.OneTimescale
using INT.Models
using DifferentiationInterface
using ABC


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

    @testset "Model Inference using ABC" begin
        # Setup synthetic data with known parameters
        true_tau = 50.0
        dt = 1.0
        T = 1000.0
        num_trials = 500
        n_lags = 50
        time = dt:dt:T
        
        # Generate synthetic data
        data = generate_ou_process(true_tau, 3.0, dt, T, num_trials)
        
        @testset "ABC Inference - ACF" begin
            model = one_timescale_model(
                data,
                time,
                :abc;
                summary_method=:acf,
                prior=[Uniform(1.0, 100.0)],
                n_lags=n_lags,
                distance_method=:linear
            )
            
            # Custom parameters for faster testing
            param_dict = get_param_dict_abc()
            param_dict[:steps] = 10
            param_dict[:max_iter] = 10000
            param_dict[:target_epsilon] = 1e-2
            
            posterior_samples, posterior_MAP, abc_results = Models.solve(model, param_dict)
            
            # Test posterior properties
            @test posterior_MAP[1] ≈ true_tau atol=10.0
            @test size(posterior_samples, 2) == 1  # One parameter (tau)
            @test !isempty(posterior_samples)
            @test !any(isnan, posterior_samples)
            @test abc_results isa ABC.ABCResults
        end
        
        @testset "ABC Inference - PSD" begin
            model = one_timescale_model(
                data,
                time,
                :abc;
                summary_method=:psd,
                prior=[Uniform(1.0, 100.0)],
                freqlims=(0.5 / 1000.0, 100.0 / 1000.0),
                distance_method=:logarithmic
            )
            
            param_dict = get_param_dict_abc()
            param_dict[:epsilon_0] = 1.0
            param_dict[:steps] = 100
            param_dict[:max_iter] = 10000
            param_dict[:target_epsilon] = 1e-4
            param_dict[:N] = 10000
            param_dict[:distance_max] = 500.0
            
            posterior_samples, posterior_MAP, abc_results = Models.solve(model, param_dict)
            
            # Test posterior properties
            @test posterior_MAP[1] ≈ true_tau atol=10.0
            @test size(posterior_samples, 2) == 1
            @test !isempty(posterior_samples)
            @test !any(isnan, posterior_samples)
            
            # Test ABC convergence
            @test abc_results.epsilon_history[end] < abc_results.epsilon_history[1]
        end
        
        # This is too slow, not recommended. Keeping here for completeness. 
        # @testset "Combined Distance Inference" begin
        #     model = one_timescale_model(
        #         data,
        #         time,
        #         :abc;
        #         summary_method=:acf,
        #         prior=[Uniform(1.0, 100.0)],
        #         n_lags=n_lags,
        #         distance_method=:linear,
        #         distance_combined=true,
        #         weights=[0.7, 0.3]
        #     )
            
        #     param_dict = get_param_dict_abc()
        #     param_dict[:epsilon_0] = 1.0
        #     param_dict[:steps] = 10
        #     param_dict[:max_iter] = 10000
        #     param_dict[:target_epsilon] = 1e-2
        #     param_dict[:N] = 10000
        #     param_dict[:distance_max] = 500.0
            
        #     posterior_samples, posterior_MAP, abc_record = Models.solve(model, param_dict)
            
        #     # Test posterior properties
        #     @test size(posterior_samples, 2) == 1
        #     @test !isempty(posterior_samples)
        #     @test !any(isnan, posterior_samples)
            
        #     # Test MAP estimate
        #     @test abs(posterior_MAP[1] - true_tau) < 5.0
        # end
    end

    
@testset "OneTimescaleModel ADVI Tests" begin
    # Set random seed for reproducibility

    # Generate test data
    true_tau = 300.0 / 1000.0  # 300ms converted to seconds
    num_trials = 30
    T = 10.0
    dt = 1 / 500
    times = dt:dt:T
    nlags = 150

    data_ts = generate_ou_process(true_tau, 1.0, dt, T, num_trials)

    @testset "Model Construction" begin
        model = one_timescale_model(
            data_ts, 
            times, 
            :advi;
            summary_method=:acf,
            prior=[Normal(0.3, 0.2)]
        )

        @test model.fit_method == :advi
        @test model.summary_method == :acf
        @test model.prior isa Normal
    end

    @testset "ADVI Fitting" begin
        model = one_timescale_model(
            data_ts, 
            times, 
            :advi;
            summary_method=:acf,
            prior=[Normal(0.3, 0.2)]
        )

        # Test with default parameters
        result = INT.solve(model)
        samples = result.samples
        map_estimate = result.MAP
        chain = result.chain
        
        @test size(samples, 2) == 4000  # Default n_samples
        @test length(map_estimate) == 2  # One parameter (tau) and one uncertainty (sigma)
        @test map_estimate[1] > 0  # Tau should be positive
        
        # Test with custom parameters
        param_dict = Dict(
            :n_samples => 2000,
            :n_iterations => 5,
            :n_elbo_samples => 10,
            :autodiff => AutoForwardDiff()
        )
        
        result2 = INT.solve(model, param_dict)
        samples2 = result2.samples
        map_estimate2 = result2.MAP
        chain2 = result2.chain

        # Truncated vs softplus (with @btime):
        # truncated: 13.895 s (1452078 allocations: 35.39 GiB)
        # softplus : 14.233 s (1658998 allocations: 35.54 GiB)
        
        @test size(samples2, 2) == 2000  # Custom n_samples
        @test length(map_estimate2) == 2
        @test map_estimate2[1] > 0
    end
    
end
end
