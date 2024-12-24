# test/test_abc.jl
using Distributions
using BayesianINT
@testset "ABC Algorithm" begin
    @testset "Basic ABC" begin
        # Create simple test model
        prior = [Uniform(0.0, 10.0)]
        true_param = 5.0
        
        # Simple model that generates normal distribution
        model = Models.BaseModel(
            randn(100),  # data
            prior,
            [0.0],      # dummy summary stat
            1.0         # epsilon
        )
        
        # Run ABC
        samples, distances = basic_abc(
            model,
            min_samples=10,
            epsilon=1.0,
            max_iter=1000
        )
        
        # Test basic properties
        @test length(samples) ≥ 10
        @test length(samples) == length(distances)
        @test all(d -> d ≤ 1.0, distances)
    end
    
    @testset "Parallel ABC" begin
        prior = [Uniform(0.0, 10.0)]
        model = Models.BaseModel(
            randn(100),
            prior,
            [0.0],
            1.0
        )
        
        # Run parallel ABC
        results = parallel_basic_abc(
            model,
            2;  # use 2 processes
            samples_per_proc=5,
            epsilon=1.0,
            max_iter=1000
        )
        
        samples, distances = results
        
        # Test basic properties
        @test length(samples) ≥ 10  # Should have at least 10 samples (5 per process)
        @test length(samples) == length(distances)
        @test all(d -> d ≤ 1.0, distances)
    end
end
