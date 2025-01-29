using Test
using Statistics
using Distributions
using BayesianINT
using BayesianINT.OrnsteinUhlenbeck
using BayesianINT.OneTimescaleWithMissing
using BayesianINT.Models
using NaNStatistics

@testset "OneTimescaleWithMissing Model Tests" begin
    # Setup test data and parameters
    dt = 0.01
    T = 100.0
    num_trials = 10
    n_lags = 3000
    ntime = Int(T / dt)
    time = 0:dt:T
    
    # Generate test data
    test_data = randn(num_trials, ntime)
    
    # Add missing values (10% of data)
    missing_mask = rand(size(test_data)...) .< 0.1
    test_data[missing_mask] .= NaN
    
    data_mean = nanmean(test_data)
    data_sd = nanstd(test_data)
    
    @testset "Model Construction - ACF and ABC" begin
        model = one_timescale_with_missing_model(
            test_data,
            time,
            :abc;
            summary_method=:acf,
            prior=[Uniform(0.1, 10.0)],  # tau prior
            n_lags=n_lags,
            distance_method=:linear
        )

        @test model isa OneTimescaleWithMissingModel
        @test model isa AbstractTimescaleModel
        @test size(model.data) == (num_trials, ntime)
        @test model.prior[1] isa Uniform
        @test model.fit_method == :abc
        @test model.summary_method == :acf
        @test model.distance_method == :linear
        @test size(model.missing_mask) == size(test_data)
        @test model.missing_mask == missing_mask
    end

    @testset "Informed Prior" begin
        model = one_timescale_with_missing_model(
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
        model = one_timescale_with_missing_model(
            test_data,
            time,
            :abc;
            summary_method=:acf,
            n_lags=100
        )
        
        theta = [1.0]  # test parameter (tau)
        simulated_data = Models.generate_data(model, theta)
        
        @test size(simulated_data) == (model.numTrials, Int(model.T/model.dt))
        @test all(isnan.(simulated_data[model.missing_mask]))
        @test !all(isnan.(simulated_data[.!model.missing_mask]))
        
        # Test statistical properties of non-missing data
        valid_data = filter(!isnan, simulated_data)
        @test abs(std(valid_data) - model.data_sd) < 0.2
        @test abs(mean(valid_data) - model.data_mean) < 0.2
    end

    @testset "summary_stats and distance_function" begin
        model = one_timescale_with_missing_model(
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
        @test all(abs.(stats1) .<= 1.0) || stats1[1] ≈ 1.0  # All autocorrelations should be ≤ 1
        
        # Test distance function
        distance = Models.distance_function(model, stats1, stats2)
        @test distance isa Float64
        @test distance >= 0.0
        @test Models.distance_function(model, stats1, stats1) ≈ 0.0 atol=1e-10
        @test distance > Models.distance_function(model, stats1, stats1)
    end

    @testset "Combined Distance" begin
        model = one_timescale_with_missing_model(
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
        true_tau = 20.0
        dt = 1.0
        T = 1000.0
        num_trials = 500
        n_lags = 50
        time = dt:dt:T
        
        # Generate synthetic data
        data = generate_ou_process(true_tau, 3.0, dt, T, num_trials)
        
        # Add missing values (10% of data)
        missing_mask = rand(size(data)...) .< 0.1
        data[missing_mask] .= NaN
        
        @testset "ABC Inference - ACF" begin
            model = one_timescale_with_missing_model(
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
            param_dict[:target_epsilon] = 1e-3
            
            posterior_samples, posterior_MAP, abc_record = Models.solve(model, param_dict)
            
            # Test posterior properties
            @test posterior_MAP[1] ≈ true_tau atol=10.0
            @test size(posterior_samples, 2) == 1  # One parameter (tau)
            @test !isempty(posterior_samples)
            @test !any(isnan, posterior_samples)
            
            # Test ABC convergence
            final_epsilon = abc_record[end].epsilon
            @test final_epsilon < abc_record[1].epsilon
            @test abc_record[end].n_accepted >= param_dict[:min_accepted]
        end
    end

    @testset "Model Behavior with Missing Data" begin
        # Test effect of different timescales with missing data
        theta1 = [0.5]  # faster timescale
        theta2 = [2.0]  # slower timescale
        
        model = one_timescale_with_missing_model(
            data,
            time,
            :abc;
            summary_method=:acf,
            n_lags=100
        )
        
        data1 = Models.generate_data(model, theta1)
        data2 = Models.generate_data(model, theta2)
        
        ac1 = Models.summary_stats(model, data1)
        ac2 = Models.summary_stats(model, data2)
        
        # Test that slower timescale has higher autocorrelation at longer lags
        lag_idx = 10  # Compare at lag 10
        @test mean(ac2[lag_idx]) > mean(ac1[lag_idx])
        
        # Test that missing data patterns are preserved
        @test all(isnan.(data1[model.missing_mask]))
        @test all(isnan.(data2[model.missing_mask]))
    end
end

@testset "OneTimescaleWithMissingModel ADVI Tests" begin
    # Setup test data
    true_tau = 300.0 / 1000.0  # 300ms converted to seconds
    num_trials = 30
    T = 10.0
    dt = 1 / 500
    times = dt:dt:T
    nlags = 150

    # Generate test data with missing values
    data_ts = generate_ou_process(true_tau, 1.0, dt, T, num_trials)
    missing_mask = rand(size(data_ts)...) .< 0.1  # 10% missing data
    data_ts[missing_mask] .= NaN

    @testset "Model Construction with ADVI" begin
        model = one_timescale_with_missing_model(
            data_ts, 
            times, 
            :advi;
            summary_method=:acf,
            prior=[Normal(0.3, 0.2)]
        )

        @test model.fit_method == :advi
        @test model.summary_method == :acf
        @test model.prior[1] isa Normal
        @test all(model.missing_mask .== missing_mask)
    end

    @testset "ADVI Fitting with Missing Data" begin
        model = one_timescale_with_missing_model(
            data_ts, 
            times, 
            :advi;
            summary_method=:acf,
            prior=[Normal(0.3, 0.2)]
        )

        # Test with default parameters
        samples, map_estimate, vi_result = Models.solve(model)
        
        @test size(samples, 2) == 4000  # Default n_samples
        @test length(map_estimate) == 2  # One parameter (tau) and uncertainty
        @test map_estimate[1] > 0  # Tau should be positivee
        
        # Test with custom parameters
        param_dict = Dict(
            :n_samples => 2000,
            :n_iterations => 5,
            :n_elbo_samples => 5,
            :optimizer => AutoForwardDiff()
        )
        
        samples2, map_estimate2, vi_result2 = Models.solve(model, param_dict)
        
        @test size(samples2, 1) == 2000  # Custom n_samples
        @test length(map_estimate2) == 1
        @test map_estimate2[1] > 0
    end

    # TODO: Figure out AutoDiff with LombScargle
    # @testset "ADVI with PSD Summary Statistics" begin
        
    #     # Test with PSD
    #     model_psd = one_timescale_with_missing_model(
    #         data_ts, 
    #         times, 
    #         :advi;
    #         summary_method=:psd,
    #         prior=[Normal(0.3, 0.2)]
    #     )
        
    #     results = Models.solve(model_psd)
        
    #     # Both methods should give reasonable results
    #     @test abs(map_acf[1] - true_tau) < 0.2
    #     @test abs(map_psd[1] - true_tau) < 0.2
        
    #     # Results should be similar between methods
    #     @test abs(map_acf[1] - map_psd[1]) < 0.1
    # end
end 