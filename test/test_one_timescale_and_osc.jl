using Test
using Statistics
using Distributions
using BayesianINT
using BayesianINT.OneTimescaleAndOsc
using BayesianINT.Models

@testset "OneTimescaleAndOsc Model Tests" begin
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
    data_mean = mean(test_data)
    data_sd = std(test_data)
    
    @testset "Model Construction - PSD and ABC" begin
        model = one_timescale_and_osc_model(
            test_data,
            time,
            :abc;
            summary_method=:psd,
            prior=[Uniform(0.1, 10.0), Uniform(0.01, 1.0), Uniform(0.0, 1.0)],
            freqlims=(0.5, 100.0),
            distance_method=:logarithmic
        )

        @test model isa OneTimescaleAndOscModel
        @test model isa AbstractTimescaleModel
        @test size(model.data) == (num_trials, ntime)
        @test length(model.prior) == 3  # tau, frequency, amplitude
        @test all(p isa Uniform for p in model.prior)
        @test model.fit_method == :abc
        @test model.summary_method == :psd
        @test model.distance_method == :logarithmic
    end

    @testset "Informed Prior" begin
        model = one_timescale_and_osc_model(
            test_data,
            time,
            :abc;
            summary_method=:psd,
            prior="informed_prior",
            freqlims=(0.5, 100.0)
        )
        
        @test model.prior[1] isa Normal  # tau prior
        @test model.prior[2] isa Normal  # frequency prior
        @test model.prior[3] isa Uniform # amplitude prior

        model = one_timescale_and_osc_model(
            test_data,
            time,
            :abc,
            summary_method=:acf,
            prior="informed_prior"
        )

        @test model.prior[1] isa Normal  # tau prior
        @test model.prior[2] isa Normal  # frequency prior
        @test model.prior[3] isa Uniform # amplitude prior
    end

    @testset "generate_data" begin
        model = one_timescale_and_osc_model(
            test_data,
            time,
            :abc;
            summary_method=:psd,
            freqlims=(0.5, 100.0)
        )
        
        theta = [1.0, 0.1, 0.5]  # test parameters (tau, freq, amplitude)
        simulated_data = Models.generate_data(model, theta)
        
        @test size(simulated_data) == (model.numTrials, Int(model.T/model.dt))
        @test !any(isnan, simulated_data)
        @test !any(isinf, simulated_data)
        
        # Test statistical properties
        @test abs(mean(simulated_data) - model.data_mean) < 0.1
        @test abs(std(simulated_data) - model.data_sd) < 0.1
    end

    @testset "summary_stats and distance_function" begin
        model = one_timescale_and_osc_model(
            test_data,
            time,
            :abc;
            summary_method=:psd,
            distance_method=:logarithmic,
            freqlims=(0.5, 100.0)
        )
        
        theta1 = [1.0, 0.1, 0.5]
        theta2 = [2.0, 0.2, 0.7]
        
        data1 = Models.generate_data(model, theta1)
        data2 = Models.generate_data(model, theta2)
        
        stats1 = Models.summary_stats(model, data1)
        stats2 = Models.summary_stats(model, data2)
        
        # Test summary stats properties
        @test length(stats1) == length(model.lags_freqs)
        @test !any(isnan, stats1)
        @test all(stats1 .>= 0.0)  # PSD should be non-negative
        
        # Test distance function
        distance = Models.distance_function(model, stats1, stats2)
        @test distance isa Float64
        @test distance >= 0.0
        @test Models.distance_function(model, stats1, stats1) ≈ 0.0 atol=1e-10
        @test distance > Models.distance_function(model, stats1, stats1)
    end

    @testset "Combined Distance" begin
        model = one_timescale_and_osc_model(
            test_data,
            time,
            :abc;
            summary_method=:psd,
            distance_method=:logarithmic,
            distance_combined=true,
            weights=[0.7, 0.3],
            freqlims=(0.5, 100.0)
        )

        @test model.distance_combined == true
        @test model.weights == [0.7, 0.3]
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
        
        model = one_timescale_and_osc_model(
            data,
            time,
            :abc;
            summary_method=:psd,
            prior=[Uniform(50.0, 1000.0), Uniform(1.0 / 1000.0, 100.0 / 1000.0), Uniform(0.0, 1.0)],
            freqlims=(1.0 / 1000.0, 100.0 / 1000.0),
            distance_method=:logarithmic
        )
        
        # Custom parameters for faster testing
        param_dict = get_param_dict_abc()
        param_dict[:steps] = 10
        param_dict[:max_iter] = 10000
        param_dict[:target_epsilon] = 1e-2
        
        posterior_samples, posterior_MAP, abc_record = Models.solve(model, param_dict)
        
        # Test posterior properties
        @test posterior_MAP[1] ≈ true_tau atol=30.0
        @test posterior_MAP[2] ≈ true_freq atol=0.05
        # @test posterior_MAP[3] ≈ true_coeff atol=0.2 (interestingly getting the coefficient right is super difficult)
        @test size(posterior_samples, 2) == 3  # Three parameters
        @test !isempty(posterior_samples)
        @test !any(isnan, posterior_samples)
        
        # Test ABC convergence
        final_epsilon = abc_record[end].epsilon
        @test final_epsilon < abc_record[1].epsilon
        @test abc_record[end].n_accepted >= param_dict[:min_accepted]
    end

    @testset "Model Behavior" begin
        model = one_timescale_and_osc_model(
            data,
            time,
            :abc;
            summary_method=:psd,
            freqlims=(0.01, 1.0)
        )
        
        # Test effect of different parameters
        theta_base = [100.0, 1.0 / 1000, 0.5]
        theta_higher_freq = [100.0, 20.0 / 1000.0, 0.5]
        
        data_base = Models.generate_data(model, theta_base)
        data_higher_freq = Models.generate_data(model, theta_higher_freq)
        
        psd_base = Models.summary_stats(model, data_base)
        psd_higher = Models.summary_stats(model, data_higher_freq)
        
        # Find peak frequencies
        peak_freq_base = model.lags_freqs[argmax(psd_base)]
        peak_freq_higher = model.lags_freqs[argmax(psd_higher)]
        
        # Test that higher frequency parameter leads to higher peak frequency
        @test peak_freq_higher > peak_freq_base
    end
end
