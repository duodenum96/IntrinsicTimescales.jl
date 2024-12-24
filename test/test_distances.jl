# test/test_distances.jl
@testset "Distance Functions" begin
    @testset "Linear Distance" begin
        x = [1.0, 2.0, 3.0]
        y = [1.1, 2.1, 3.1]
        
        d = linear_distance(x, y)
        
        # Test basic properties
        @test d ≥ 0
        @test linear_distance(x, x) ≈ 0
        @test linear_distance(x, y) ≈ linear_distance(y, x)
        
        # Test with known difference
        @test d ≈ 0.01 atol=1e-10
    end
    
    @testset "Logarithmic Distance" begin
        x = [1.0, 2.0, 3.0]
        y = [1.1, 2.2, 3.3]
        
        d = logarithmic_distance(x, y)
        
        # Test basic properties
        @test d ≥ 0
        @test logarithmic_distance(x, x) ≈ 0
        @test logarithmic_distance(x, y) ≈ logarithmic_distance(y, x)
        
        # Test with known values
        expected = mean((log.(x) .- log.(y)).^2)
        @test d ≈ expected
    end
end
