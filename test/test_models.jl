# test/test_models.jl
using Distributions
@testset "Two Timescale Model" begin
    @testset "Parameter Generation" begin
        prior = [
            Uniform(0.0, 60.0),
            Uniform(20.0, 140.0),
            Uniform(0.0, 1.0)
        ]
        
        model = TwoTimescaleModel(
            randn(10, 100),  # dummy data
            prior,
            zeros(10),      # dummy summary stats
            1.0,            # epsilon
            1.0,            # deltaT
            1.0,            # binSize
            100.0,          # T
            10,             # numTrials
            0.0,            # data_mean
            1.0             # data_var
        )
        
        # Test parameter drawing
        theta = Models.draw_theta(model)
        @test length(theta) == 3
        @test 0 ≤ theta[1] ≤ 60
        @test 20 ≤ theta[2] ≤ 140
        @test 0 ≤ theta[3] ≤ 1
        
        # Test data generation
        data = Models.generate_data(model, theta)
        @test size(data) == (10, 100)
        @test abs(mean(data) - model.data_mean) < 0.1
        @test abs(var(data) - model.data_var) < 0.1
    end
end