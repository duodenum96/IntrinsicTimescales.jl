using Test
using Statistics
using Distributions
using INT
using INT.OrnsteinUhlenbeck
using INT.OneTimescaleAndOscWithMissing
using INT.Models
using NaNStatistics

@testset "OneTimescaleAndOscWithMissing Model Tests" begin
    # Setup test data and parameters
    dt = 0.01
    T = 100.0
    num_trials = 10
    ntime = Int(T / dt)
    time = dt:dt:T
    
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
    
    data_mean = mean(skipmissing(test_data))
    data_sd = std(skipmissing(test_data))
    
    @testset "Model Construction - PSD and ABC" begin
        model = one_timescale_and_osc_with_missing_model(
            test_data,
            time,
            :abc;
            summary_method=:psd,
            prior=[Uniform(0.1, 10.0), Uniform(0.01, 1.0), Uniform(0.0, 1.0)],
            freqlims=(0.5, 100.0),
            distance_method=:logarithmic
        )

        @test model isa OneTimescaleAndOscWithMissingModel
        @test model isa AbstractTimescaleModel
        @test size(model.data) == (num_trials, ntime)
        @test length(model.prior) == 3  # tau, frequency, amplitude
        @test all(p isa Uniform for p in model.prior)
        @test model.fit_method == :abc
        @test model.summary_method == :psd
        @test model.distance_method == :logarithmic
        @test size(model.missing_mask) == size(test_data)
    end

    @testset "Informed Prior" begin
        model = one_timescale_and_osc_with_missing_model(
            test_data,
            time,
            :abc;
            summary_method=:psd,
            prior="informed_prior",
            freqlims=(0.5, 100.0),
        )
        
        @test model.prior[1] isa Normal  # tau prior
        @test model.prior[2] isa Normal  # frequency prior
        @test model.prior[3] isa Uniform # amplitude prior
    end

    @testset "generate_data" begin
        dt = 1.0
        T = 300.0
        time = dt:dt:T
        num_trials = 15
        ntime = Int(T / dt)
        true_tau = 20.0
        true_freq = 0.01
        true_coeff = 0.5
        test_data = generate_ou_with_oscillation(
            [true_tau, true_freq, true_coeff],
            dt, T, num_trials, 0.0, 1.0
        )
        model = one_timescale_and_osc_with_missing_model(
            test_data,
            time,
            :abc;
            summary_method=:psd,
            freqlims=(0.5 / 1000.0, 100.0 / 1000.0)
        )
        
        theta = [true_tau, true_freq, true_coeff]
        simulated_data = Models.generate_data(model, theta)
        
        @test size(simulated_data) == size(test_data)
        @test all(isnan.(simulated_data[model.missing_mask]))
        @test !all(isnan.(simulated_data[.!model.missing_mask]))
    end

    @testset "Model Inference using ABC" begin
        # Generate synthetic data with known parameters
        true_tau = 300.0
        true_freq = 10.0 / 1000.0
        true_coeff = 0.5
        dt = 1.0
        T = 20.0 * 1000.0
        num_trials = 15
        
        data = generate_ou_with_oscillation(
            [true_tau, true_freq, true_coeff],
            dt, T, num_trials, 0.0, 1.0
        )
        time = dt:dt:T
        
        # Add missing values
        missing_mask = rand(size(data)...) .< 0.1
        data[missing_mask] .= NaN
        
        model = one_timescale_and_osc_with_missing_model(
            data,
            time,
            :abc;
            summary_method=:psd,
            prior=[Uniform(50.0, 1000.0), Uniform(1.0/1000.0, 100.0/1000.0), Uniform(0.0, 1.0)],
            freqlims=(1.0/1000.0, 100.0/1000.0),
            distance_method=:logarithmic,
        )
        
        # Custom parameters for faster testing
        param_dict = get_param_dict_abc()
        param_dict[:steps] = 3
        param_dict[:max_iter] = 10000
        param_dict[:distance_max] = 100.0
        param_dict[:target_epsilon] = 1e-1
        
        results = Models.fit(model, param_dict)
        samples = results.final_theta
        map_estimate = results.MAP
        
        # Test posterior properties
        @test map_estimate[1] isa Float64
        @test map_estimate[2] isa Float64

        @test size(samples, 2) == 3
        @test !isempty(samples)
        @test !any(isnan, samples)
        

        # Test ABC convergence
        @test results.epsilon_history[end] < results.epsilon_history[1]
        @test length(results.theta_history[end]) >= param_dict[:min_accepted]
    end


    @testset "Model Behavior with Missing Data" begin
        dt = 1.0
        T = 500.0
        time = dt:dt:T
        num_trials = 15
        ntime = Int(T / dt)
        true_tau = 20.0
        true_freq = 0.01
        true_coeff = 0.5
        test_data = generate_ou_with_oscillation(
            [true_tau, true_freq, true_coeff],
            dt, T, num_trials, 0.0, 1.0
        )
        missing_mask = rand(size(test_data)...) .< 0.1
        test_data[missing_mask] .= NaN
        
        model = one_timescale_and_osc_with_missing_model(
            test_data,
            time,
            :abc;
            summary_method=:psd,
            freqlims=(0.001, 0.1)
        )
        
        # Test effect of different parameters
        theta_base = [10.0, 0.01, 0.5]
        theta_higher_freq = [10.0, 0.05, 0.5]
        
        data_base = Models.generate_data(model, theta_base)
        data_higher_freq = Models.generate_data(model, theta_higher_freq)
        
        # Verify missing data patterns
        @test all(isnan.(data_base[model.missing_mask]))
        @test all(isnan.(data_higher_freq[model.missing_mask]))
        
        # Test summary statistics with missing data
        stats_base = Models.summary_stats(model, data_base)
        stats_higher = Models.summary_stats(model, data_higher_freq)

        amp_base, knee_base = lorentzian_initial_guess(stats_base, model.lags_freqs)
        amp_higher, knee_higher = lorentzian_initial_guess(stats_higher, model.lags_freqs)
        lorentzian_base = lorentzian(model.lags_freqs, [amp_base, knee_base])
        lorentzian_higher = lorentzian(model.lags_freqs, [amp_higher, knee_higher])

        residual_base = stats_base .- lorentzian_base
        residual_higher = stats_higher .- lorentzian_higher
        
        # Find peak frequencies
        peak_freq_base = find_oscillation_peak(residual_base, model.lags_freqs, min_freq=model.lags_freqs[1], max_freq=model.lags_freqs[end])
        peak_freq_higher = find_oscillation_peak(residual_higher, model.lags_freqs, min_freq=model.lags_freqs[1], max_freq=model.lags_freqs[end])
        
        # Test that higher frequency parameter leads to higher peak frequency
        @test peak_freq_higher > peak_freq_base
    end

    @testset "Distance Function with Missing Data" begin
        # Setup small test case
        dt = 1.0
        T = 1000.0
        num_trials = 5
        time = dt:dt:T
        
        # Generate two datasets with different parameters
        theta1 = [20.0, 0.01, 0.5]  # slow oscillation
        theta2 = [20.0, 0.05, 0.5]  # faster oscillation
        
        data1 = generate_ou_with_oscillation(theta1, dt, T, num_trials, 0.0, 1.0)
        data2 = generate_ou_with_oscillation(theta2, dt, T, num_trials, 0.0, 1.0)
        
        # Create missing mask
        missing_mask = rand(size(data1)...) .< 0.1
        data1[missing_mask] .= NaN
        data2[missing_mask] .= NaN
        
        # Test with PSD
        model_psd = one_timescale_and_osc_with_missing_model(
            data1,
            time,
            :abc;
            summary_method=:psd,
            freqlims=(0.001, 0.1),
            distance_method=:logarithmic
        )
        
        # Get summary statistics
        stats1_psd = Models.summary_stats(model_psd, data1)
        stats2_psd = Models.summary_stats(model_psd, data2)
        
        # Test distance calculation
        d1 = Models.distance_function(model_psd, stats1_psd, stats1_psd)
        d2 = Models.distance_function(model_psd, stats1_psd, stats2_psd)
        
        @test d1 ≈ 0.0 atol=1e-10  # Distance to self should be ~0
        @test d2 > d1              # Different signals should have positive distance
        
        # Test with ACF
        model_acf = one_timescale_and_osc_with_missing_model(
            data1,
            time,
            :abc;
            summary_method=:acf,
            n_lags=100,
            distance_method=:linear
        )
        
        # Get summary statistics
        stats1_acf = Models.summary_stats(model_acf, data1)
        stats2_acf = Models.summary_stats(model_acf, data2)
        
        # Test distance calculation
        d1_acf = Models.distance_function(model_acf, stats1_acf, stats1_acf)
        d2_acf = Models.distance_function(model_acf, stats1_acf, stats2_acf)
        
        @test d1_acf ≈ 0.0 atol=1e-10  # Distance to self should be ~0
        @test d2_acf > d1_acf          # Different signals should have positive distance
        
        # Test combined distance
        model_combined = one_timescale_and_osc_with_missing_model(
            data1,
            time,
            :abc;
            summary_method=:psd,
            freqlims=(0.001, 0.1),
            distance_method=:logarithmic,
            distance_combined=true,
            weights=[0.4, 0.3, 0.3]
        )
        

        # Get summary statistics
        stats1_combined = Models.summary_stats(model_combined, data1)
        stats2_combined = Models.summary_stats(model_combined, data2)
        
        # Test combined distance calculation
        d1_combined = Models.distance_function(model_combined, stats1_combined, stats1_combined)
        d2_combined = Models.distance_function(model_combined, stats1_combined, stats2_combined)
        
        @test d1_combined ≈ 0.0 atol=1e-10  # Distance to self should be ~0
        @test d2_combined > d1_combined      # Different signals should have positive distance
    end
end

@testset "OneTimescaleAndOscWithMissing ADVI Tests" begin
    # Setup test data
    true_tau = 300.0 / 1000.0  # 300ms converted to seconds
    true_freq = 40.0  # Hz
    true_coeff = 0.5
    true_params = [true_tau, true_freq, true_coeff]
    
    num_trials = 30
    T = 10.0
    dt = 1 / 500
    times = dt:dt:T
    
    # Generate test data with oscillation and missing values
    data_ts = generate_ou_with_oscillation(true_params, dt, T, num_trials, 0.0, 1.0)
    missing_mask = rand(size(data_ts)...) .< 0.1
    data_ts[missing_mask] .= NaN

    @testset "Model Construction with ADVI" begin
        model = one_timescale_and_osc_with_missing_model(
            data_ts, 
            times, 
            :advi;
            summary_method=:psd,
            prior=[Normal(0.3, 0.2), Normal(40.0, 5.0), Uniform(0.0, 1.0)]
        )

        @test model.fit_method == :advi
        @test model.summary_method == :psd
        @test length(model.prior) == 3
        @test all(model.missing_mask .== missing_mask)
    end

    @testset "ADVI Fitting with ACF" begin
        model = one_timescale_and_osc_with_missing_model(
            data_ts, 
            times, 
            :advi;
            summary_method=:acf,
            prior=[Normal(0.3, 0.2), Normal(40.0, 5.0), Uniform(0.0, 1.0)]
        )

        # Test with default parameters
        param_dict = get_param_dict_advi()
        param_dict[:n_samples] = 2000
        param_dict[:n_iterations] = 2
        param_dict[:n_elbo_samples] = 3
        param_dict[:autodiff] = AutoForwardDiff()

        results = Models.fit(model, param_dict)
        samples = results.samples
        map_estimate = results.MAP
        variational_posterior = results.variational_posterior
        

        @test size(samples, 2) == 2000  # Default n_samples
        @test length(map_estimate) == 4  # Three parameters + sigma
        @test map_estimate[1] > 0  # Tau should be positive
        @test map_estimate[2] > 0  # Frequency should be positive
        @test 0 <= map_estimate[3] <= 1  # Coefficient should be between 0 and 1 
    end
end 

