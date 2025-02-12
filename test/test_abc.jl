using Test
using Distributions
using IntrinsicTimescales
using Statistics
using StatsBase
using LinearAlgebra

@testset "ABC Module" begin
    @testset "Basic ABC" begin
        # Create simple test model
        prior = [Uniform(0.0, 10.0)]
        true_param = 5.0
        
        # Simple model that generates normal distribution
        model = Models.BaseModel(
            randn(100),  # data
            [1.0],
            [0.0],      # dummy summary stat
            :abc,
            :psd,
            [0.0],
            prior,
            :acw0,
            :linear,
            1.0,
            1.0,         # epsilon
            1.0,
            1.0,
            1.0
        )
        
        # Run ABC
        result = ABC.basic_abc(
            model,
            epsilon=1.0,
            max_iter=1000,
            min_accepted=10
        )
        
        # Test basic properties
        @test length(result.theta_accepted) ≥ 10
        @test length(result.distances) == length(result.theta_accepted)
        @test all(d -> d ≤ 1.0, result.distances)
        @test result.n_accepted ≥ 10
        @test result.n_total ≤ 1000
    end
    
    @testset "Helper Functions" begin
        @testset "Weighted Covariance" begin
            # Test 1D case
            x = [1.0 2.0; 3.0 4.0; 5.0 6.0]
            w = [0.2, 0.3, 0.5]
            covar = ABC.weighted_covar(x, w)
            @test covar isa Matrix{Float64}
            @test all(covar .> 0)
            
            # Test 2D case
            X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
            covar = ABC.weighted_covar(X, w)
            @test size(covar) == (2, 2)
            @test issymmetric(covar)
        end
        
        @testset "Effective Sample Size" begin
            w = [0.2, 0.3, 0.5]
            ess = ABC.effective_sample_size(w)
            @test 1 ≤ ess ≤ length(w)
            
            # Equal weights should give maximum ESS
            w_equal = fill(1/3, 3)
            ess_equal = ABC.effective_sample_size(w_equal)
            @test ess_equal ≈ 3.0
        end
        
        @testset "Weight Calculation" begin
            theta_prev = [1.0 2.0; 3.0 4.0]
            theta = [1.5 2.5; 3.5 4.5]
            tau_squared = [0.1 0.0; 0.0 0.1]
            weights = [0.5, 0.5]
            prior = [Uniform(0.0, 5.0), Uniform(0.0, 5.0)]
            
            new_weights = ABC.calc_weights(theta_prev, theta, tau_squared, weights, prior)
            
            @test length(new_weights) == size(theta, 2)
            @test all(w -> 0 ≤ w ≤ 1, new_weights)
            @test sum(new_weights) ≈ 1.0
        end
        
        @testset "MAP Estimation" begin
            # Create test data from a known distribution
            true_params = [3.0, 5.0]
            n_samples = 1000
            theta_accepted = zeros(n_samples, 2)
            theta_accepted[:,1] = rand(Normal(true_params[1], 0.5), n_samples)
            theta_accepted[:,2] = rand(Normal(true_params[2], 0.5), n_samples)
            
            # Test MAP estimation
            N = 1000
            theta_map = ABC.find_MAP(theta_accepted, N)
            
            # Basic tests
            @test length(theta_map) == 2
            @test all(isfinite.(theta_map))
            
            # MAP estimates should be close to true parameters
            @test all(isapprox.(theta_map, true_params, atol=1.0))
            
            # Test with different N values
            theta_map_small = ABC.find_MAP(theta_accepted, 100)
            @test length(theta_map_small) == 2
        end
    end
    
    @testset "PMC ABC" begin
        # Create simple test model
        prior = [Uniform(0.0, 10.0)]
        model = Models.BaseModel(
            randn(100),  # data
            [1.0],
            [0.0],      # dummy summary stat
            :abc,
            :psd,
            [0.0],
            prior,
            :acw0,
            :linear,
            1.0,
            1.0,         # epsilon
            1.0,
            1.0,
            1.0
        )
        
        
        # Run PMC-ABC with minimal steps
        results = ABC.pmc_abc(
            model;
            epsilon_0=1.0,
            min_accepted=10,
            max_iter=10,
            steps=2,
            minAccRate=0.001
        )
        
        # Test basic properties
        @test results isa ABC.ABCResults
        @test length(results.theta_history) ≤ 2  # May stop early due to acceptance rate
        @test length(results.theta_history[1]) ≥ 10
        @test results.epsilon_history[1] ≥ results.epsilon_history[end]  # Epsilon should decrease
        @test all(i -> length(results.weights_history[i]) == size(results.theta_history[i], 1), 1:length(results.theta_history))
    end
end
