# test/test_models.jl
using Distributions
using INT
using Test
# @testset "Two Timescale Model" begin
#     @testset "Parameter Generation" begin
#         prior = [
#             Uniform(0.0, 60.0), # tau1
#             Uniform(20.0, 140.0), # tau2
#             Uniform(0.0, 1.0) # p
#         ]
        
#         model = TwoTimescaleModel(
#             randn(10, 100),  # dummy data
#             prior,
#             zeros(10),      # dummy summary stats
#             1.0,            # epsilon
#             1.0,            # dt
#             100.0,          # T
#             10,             # numTrials
#             1.0,             # data_var
#             10              # n_lags
#         )
        
#         # Test parameter drawing
#         theta = Models.draw_theta(model)
#         @test length(theta) == 3
#         @test 0 ≤ theta[1] ≤ 60
#         @test 20 ≤ theta[2] ≤ 140
#         @test 0 ≤ theta[3] ≤ 1
        
#         # Test data generation
#         data = Models.generate_data(model, theta)
#         @test size(data) == (10, 100)
#         # @test abs(mean(data) - model.data_mean) < 0.1
#         # @test abs(var(data) - model.data_var) < 0.1
#     end
# end

@testset "OneTimescaleAndOscWithMissing Model" begin
    @testset "Parameter Generation and Data" begin
        prior = [
            Uniform(0.0, 60.0),  # tau
            Uniform(0.0, 50.0),  # freq
            Uniform(0.0, 1.0)    # amplitude
        ]
        
        # Create dummy data with some missing values
        data = randn(10, 100)
        data[1:2, 1:10] .= NaN  # Add some missing values
        times = collect(range(0, 100, length=100))
        missing_mask = isnan.(data)
        dt = 1.0

        # Create summary statistics for missing data
        data_sum_stats = comp_psd_lombscargle(times, data, missing_mask, dt)
        
        model = OneTimescaleAndOscWithMissingModel(
            data,            # data with missing values
            times,          # time points
            prior,          # parameter priors
            data_sum_stats, # summary statistics
            1.0,            # epsilon
            1.0,            # dt
            100.0,          # T
            10,             # numTrials
            0.0,            # data_mean
            1.0,             # data_var
            missing_mask    # missing_mask
        )
        
        # Test parameter drawing
        theta = Models.draw_theta(model)
        @test length(theta) == 3
        @test 0 ≤ theta[1] ≤ 60    # tau
        @test 0 ≤ theta[2] ≤ 50    # freq
        @test 0 ≤ theta[3] ≤ 1     # amplitude
        
        # Test data generation
        synth_data = Models.generate_data(model, theta)
        @test size(synth_data) == size(data)
        @test any(isnan.(synth_data))  # Should have missing values
        @test all(isnan.(synth_data) .== model.missing_mask)  # Missing values should match mask
        
        # Test summary statistics
        sum_stats = Models.summary_stats(model, synth_data)
        @test length(sum_stats) == 2  # Should return PSD and frequencies
        @test length(sum_stats[1]) > 0  # PSD should not be empty
        @test length(sum_stats[2]) > 0  # Frequencies should not be empty

        # Test distance calculation
        distance = Models.distance_function(model, sum_stats, data_sum_stats)
        @test distance isa Float64
        @test distance > 0
    end
end

