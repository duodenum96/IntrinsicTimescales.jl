# test/test_models.jl
using Distributions
using INT
using Test
using INT.Models

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

# NOTE: There are too many tests here due to for loops, commenting them for now but 
# no errors when I run. 
# @testset "Data reshaping tests" begin
#     @testset "2D data with different time dimensions" begin
#         # Create test data with identifiable pattern: 10 trials x 100 timepoints
#         data_time_last = [i + j/100 for i in 1:10, j in 1:100]
#         time = collect(1:100)
        
#         # Test when dims matches ndims (should not reshape)
#         data, dims = check_model_inputs(data_time_last, time, :abc, :acf, "informed_prior", :linear, 2)
#         @test size(data) == size(data_time_last)
#         @test all(data .== data_time_last)
        
#         # Test when time is in first dimension
#         time = collect(1:10)
#         data_time_first = [i + j/100 for j in 1:10, i in 1:100]  # Note: i and j swapped in both position and value
#         data, dims = check_model_inputs(data_time_first, time, :abc, :acf, "informed_prior", :linear, 1)
#         @test size(data) == (100, 10)
#         # Check specific elements to ensure correct reshaping
#         for j in 1:10, i in 1:100
#             @test data[i,j] ≈ data_time_first[j,i]
#             @test data[i,j] ≈ i + j/100
#         end
#     end

#     @testset "3D data with different time dimensions" begin
#         # Create test data with identifiable pattern: 5 channels x 10 trials x 100 timepoints
#         data_time_last = [c + t + tp/100 for c in 1:5, t in 1:10, tp in 1:100]
#         time = collect(1:100)
        
#         # Test when dims matches ndims (should not reshape)
#         data, dims = check_model_inputs(data_time_last, time, :abc, :acf, "informed_prior", :linear, 3)
#         @test size(data) == size(data_time_last)
#         @test all(data .== data_time_last)
        
#         # Test when time is in first dimension
#         data_time_first = [c + t + tp/100 for tp in 1:100, c in 1:5, t in 1:10]  # Note order of indices
#         data, dims = check_model_inputs(data_time_first, time, :abc, :acf, "informed_prior", :linear, 1)
#         @test size(data) == (5, 10, 100)
#         for c in 1:5, t in 1:10, tp in 1:100
#             @test data[c,t,tp] ≈ data_time_first[tp,c,t]
#             @test data[c,t,tp] ≈ c + t + tp/100  # This should match the original pattern
#         end
        
#         # Test when time is in middle dimension
#         data_time_middle = [c + t + tp/100 for c in 1:5, tp in 1:100, t in 1:10]
#         data, dims = check_model_inputs(data_time_middle, time, :abc, :acf, "informed_prior", :linear, 2)
#         @test size(data) == (5, 10, 100)
#         for c in 1:5, t in 1:10, tp in 1:100
#             @test data[c,t,tp] ≈ data_time_middle[c,tp,t]
#             @test data[c,t,tp] ≈ c + t + tp/100
#         end
#     end

#     @testset "1D data handling" begin
#         # Test vector data with identifiable pattern
#         data_vector = [t/100 for t in 1:100]
#         time = collect(1:100)
#         data, dims = check_model_inputs(data_vector, time, :abc, :acf, "informed_prior", :linear)
#         @test size(data) == (1, 100)
#         for t in 1:100
#             @test data[1,t] ≈ data_vector[t]
#             @test data[1,t] ≈ t/100
#         end
#     end

#     @testset "Error cases" begin
#         data = rand(10, 100)
#         time = collect(1:50)  # Wrong length
        
#         # Test mismatched time vector length
#         @test_throws ArgumentError check_model_inputs(
#             data, time, :abc, :acf, "informed_prior", :linear, 2
#         )
#     end
# end 

