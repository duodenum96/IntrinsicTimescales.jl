using Test
using Distributions
using ABCdeZ
using IntrinsicTimescales
using Statistics
using Random

@testset "ABCdeZ Extension Tests" begin
    @testset "Parameter Dictionary Functions" begin
        @testset "get_param_dict_abcdemc" begin
            param_dict = IntrinsicTimescales.get_param_dict_abcdemc()

            # Test that it returns a dictionary
            @test param_dict isa Dict

            # Test all required keys are present
            required_keys = [:nparticles, :generations, :verbose, :rng, :parallel]
            for key in required_keys
                @test haskey(param_dict, key)
            end
        end

        @testset "get_param_dict_abcdesmc" begin
            param_dict = IntrinsicTimescales.get_param_dict_abcdesmc()

            # Test that it returns a dictionary
            @test param_dict isa Dict

            # Test all required keys are present
            required_keys = [:nparticles, :α, :δess, :nsims_max, :Kmcmc, :Kmcmc_min,
                :ABCk, :facc_min, :facc_tune, :verbose, :verboseout, :rng, :parallel]
            for key in required_keys
                @test haskey(param_dict, key)
            end
        end
    end

    @testset "ABCdeZ Inference Functions" begin
        # Create a simple test model for testing inference functions        
        true_tau = 50.0
        dt = 1.0
        T = 1000.0
        num_trials = 20
        n_lags = 50
        time = dt:dt:T
        
        # Generate synthetic data
        data = generate_ou_process(true_tau, 3.0, dt, T, num_trials)
        
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
        # Test with valid inputs
        ϵ_target = 0.01
        param_dict_mc = IntrinsicTimescales.get_param_dict_abcdemc()
        param_dict_smc = IntrinsicTimescales.get_param_dict_abcdesmc()

        # Test error on invalid method
        @test_throws Exception IntrinsicTimescales.abcdez_inference(model, ϵ_target,
                                                                    param_dict_mc,
                                                                   :invalid_method)

        results_mc, posterior_mc = IntrinsicTimescales.abcdez_inference(model, ϵ_target, param_dict_mc, :abcdemc)
        results_smc, posterior_smc = IntrinsicTimescales.abcdez_inference(model, ϵ_target, param_dict_smc, :abcdesmc)
        @test isapprox(mean(posterior_mc), true_tau)
        @test isapprox(mean(posterior_smc), true_tau)

    end
end
