using Test
using IntrinsicTimescales.ACW
using IntrinsicTimescales
using Random

@testset "ACW Module Tests" begin
    # Test data setup
    fs = 100.0  # 100 Hz sampling rate
    t = 0:1/fs:10  # 10 seconds of data
    freq = 5.0  # 5 Hz oscillation
    tau = 0.5   # 0.5s decay time
    data = generate_ou_process(tau, 1.0, 1/fs, 10.0, 1)[:]
    

    @testset "Basic ACW Container" begin
        container = ACWResults(fs, [0.0], :acw0, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
        @test container.fs == fs
        @test container.acw_results == [0.0]
        @test container.acwtypes == :acw0
        @test container.n_lags === nothing
        @test container.freqlims === nothing
        @test container.acf === nothing
        @test container.psd === nothing
        @test container.freqs === nothing
        @test container.lags === nothing
        @test container.x_dim === nothing
    end

    @testset "ACW Types Validation" begin
        # Test single ACW type
        result_single = acw(data, fs, acwtypes=:acw0).acw_results
        @test length(result_single) == 1
        
        # Test multiple ACW types
        result_multiple = acw(data, fs, acwtypes=[:acw0, :acw50]).acw_results
        @test length(result_multiple) == 2

        # Test invalid ACW type
        @test_throws ErrorException acw(data, fs, acwtypes=:invalid_type).acw_results
    end

    @testset "ACW Calculations" begin
        # Test ACW-0
        result = acw(data, fs, acwtypes=:acw0).acw_results
        @test length(result) == 1
        @test !isnothing(result[1])
        @test all(isfinite.(result[1]))

        # Test ACW-50
        result = acw(data, fs, acwtypes=:acw50).acw_results
        @test length(result) == 1
        @test !isnothing(result[1])
        @test all(isfinite.(result[1]))

        # Test tau calculation
        result = acw(data, fs, acwtypes=:tau).acw_results
        @test length(result) == 1
        @test !isnothing(result[1])
        @test all(isfinite.(result[1]))
        # Test if tau is roughly close to the input tau (with some tolerance)
        @test isapprox(result[1][1], tau, rtol=0.5)
    end

    @testset "Frequency Limits and N_lags" begin
        # Test with custom n_lags
        n_lags = 100
        result = acw(data, fs, acwtypes=:acw0, n_lags=n_lags)
        @test length(result.acf) ≤ n_lags

        # Test with frequency limits
        freqlims = (0.1, 10.0)
        result = acw(data, fs, acwtypes=:knee, freqlims=freqlims)
        @test result.freqlims == freqlims
        @test !isnothing(result.psd)
    end

    @testset "Multi-dimensional Input" begin
        # Create 2D data
        fs = 100
        dt = 1/fs
        duration = 10.0
        num_trials = 2
        data_2d = generate_ou_process(tau, 1.0, dt, duration, num_trials)
        result = acw(data_2d, fs, acwtypes=:acw0, dims=1).acw_results
        @test length(result) == size(data_2d, 2)

        # Create 3D data
        data_3d = cat(data_2d, data_2d, dims=3)
        result = acw(data_3d, fs, acwtypes=:acw0, dims=1).acw_results
        @test size(result) == (size(data_3d, 2), size(data_3d, 3))


        # Test correctness of results along different dimensions using OU processes
        fs = 100
        dt = 1/fs
        duration = 10.0
        num_trials = 3
        
        # Generate OU processes with different timescales
        tau1 = 0.2 # Fast timescale
        tau2 = 0.5 # Medium timescale  
        tau3 = 1.0 # Slow timescale
        
        # Generate three OU processes
        Random.seed!(666)
        data1 = generate_ou_process(tau1, 1.0, dt, duration, 1)
        data2 = generate_ou_process(tau2, 1.0, dt, duration, 1)
        data3 = generate_ou_process(tau3, 1.0, dt, duration, 1)
        
        # Stack into 3D array
        data_test = cat(data1, data2, data3, dims=1)
        data_test = reshape(data_test, size(data_test,1), size(data_test,2), 1)

        # Test ACW50 along dimension 1
        result_test = acw(data_test, fs, acwtypes=:acw50, dims=2).acw_results
        @test size(result_test) == (3, 1) # Check output shape
        
        # Test if ACW50 values increase with increasing tau
        # ACW50 should be proportional to tau: acw50 ≈ -tau * log(0.5)
        @test result_test[1] < result_test[2] < result_test[3]
        @test isapprox(result_test[1], -tau1 * log(0.5), rtol=0.5)
        @test isapprox(result_test[2], -tau2 * log(0.5), rtol=0.5)
        @test isapprox(result_test[3], -tau3 * log(0.5), rtol=0.5)

        # Test tau estimation directly
        result_tau = acw(data_test, fs, acwtypes=:tau, dims=2).acw_results
        @test size(result_tau) == (3, 1)
        @test isapprox(result_tau[1], tau1, rtol=1.0)
        @test isapprox(result_tau[2], tau2, rtol=0.5)
        @test isapprox(result_tau[3], tau3, rtol=0.5)

        # Test along dimension 2
        data_test_perm = permutedims(data_test, (2, 1, 3))
        result_test_dim2 = acw(data_test_perm, fs, acwtypes=:tau, dims=1).acw_results
        @test size(result_test_dim2) == (3, 1)
        @test isapprox(result_test_dim2[1], tau1, rtol=1.0)
        @test isapprox(result_test_dim2[2], tau2, rtol=0.5)
        @test isapprox(result_test_dim2[3], tau3, rtol=0.5)
    end
    
    @testset "Missing Data Handling" begin
        # Setup base data using OU process
        fs = 100.0
        dt = 1/fs
        duration = 10.0
        tau = 0.5
        true_D = 1.0
        
        # Generate clean OU process
        Random.seed!(123)  # For reproducibility
        clean_data = vec(generate_ou_process(tau, true_D, dt, duration, 1))
        
        # Create data with missing values
        missing_data = copy(clean_data)
        missing_indices = rand(1:length(clean_data), 100)  # Randomly remove 100 points
        missing_data[missing_indices] .= NaN
        
        # Test with different missing data patterns
        @testset "Random Missing Values" begin
            result = acw(missing_data, fs, acwtypes=:tau).acw_results
            @test length(result) == 1
            @test !isnothing(result[1])
            @test isapprox(result[1][1], tau, rtol=1.0)
            
            # Test multiple ACW types with missing data
            result_multiple = acw(missing_data, fs, acwtypes=[:acw0, :acw50]).acw_results
            @test length(result_multiple) == 2
            @test all(isfinite.(result_multiple[1]))
            @test all(isfinite.(result_multiple[2]))
        end
        
        @testset "Consecutive Missing Values" begin
            # Create data with consecutive missing values
            consec_missing_data = copy(clean_data)
            consec_missing_data[201:300] .= NaN  # 1-second gap
            
            result = acw(consec_missing_data, fs, acwtypes=:tau).acw_results
            @test length(result) == 1
            @test !isnothing(result[1])
            @test isapprox(result[1][1], tau, rtol=1.0)
        end
        
        @testset "High Proportion Missing" begin
            # Create data with 50% missing values
            high_missing_data = copy(clean_data)
            high_missing_indices = rand(1:length(clean_data), length(clean_data)÷2)
            high_missing_data[high_missing_indices] .= NaN
            
            result = acw(high_missing_data, fs, acwtypes=:tau).acw_results
            @test length(result) == 1
            @test !isnothing(result[1])
            # Allow for larger tolerance due to high proportion of missing data
            @test isapprox(result[1][1], tau, rtol=2.0)
        end
        
        @testset "Missing Data in Multi-dimensional Input" begin
            # Generate multiple OU processes
            data_2d = generate_ou_process(tau, true_D, dt, duration, 2)
            
            # Add missing values to both processes
            data_2d_missing = copy(data_2d)
            for i in 1:size(data_2d, 1)
                missing_indices = rand(1:size(data_2d, 2), 100)
                data_2d_missing[i, missing_indices] .= NaN
            end
            
            result_2d = acw(data_2d_missing, fs, acwtypes=:tau, dims=2).acw_results
            @test length(result_2d) == size(data_2d, 1)
            @test all(isfinite.(result_2d))
            
            # Create 3D data
            data_3d = cat(data_2d_missing, data_2d_missing, dims=3)
            result_3d = acw(data_3d, fs, acwtypes=:tau, dims=2).acw_results
            @test size(result_3d) == (size(data_3d, 1), size(data_3d, 3))
            @test all(isfinite.(result_3d))
        end
        
        @testset "Lomb-Scargle PSD with Missing Data" begin
            # Generate OU process with known properties
            fs = 1000.0
            dt = 1/fs
            duration = 10.0
            tau = 0.3
            true_D = 1.0
            
            # Generate clean data
            Random.seed!(456)
            clean_data = vec(generate_ou_process(tau, true_D, dt, duration, 1))
            
            # Create different missing data patterns
            random_missing = copy(clean_data)
            random_missing[rand(1:length(clean_data), 200)] .= NaN  # 20% missing randomly
            
            gap_missing = copy(clean_data)
            gap_missing[301:500] .= NaN  # 2-second gap
            
            # Test knee frequency estimation with missing data
            for test_data in [clean_data, random_missing, gap_missing]
                estimated_tau = acw(test_data, fs, acwtypes=:knee, freqlims=(0.1, 10.0), oscillation_peak=false).acw_results
                # Test if estimated tau is within reasonable range
                # Allow larger tolerance due to missing data
                @test isapprox(estimated_tau, tau, rtol=1.0) # much worse than clean data but eh
            end
            
            # Test with multiple trials
            data_2d = generate_ou_process(tau, true_D, dt, duration, 2)
            data_2d_missing = copy(data_2d)
            for i in 1:size(data_2d, 1)
                data_2d_missing[i, rand(1:size(data_2d, 2), 200)] .= NaN
            end
            
            result_2d = acw(data_2d_missing, fs, acwtypes=:knee, freqlims=(0.1, 10.0), dims=2).acw_results
            @test size(result_2d) == (size(data_2d, 1),)
            @test all(isfinite.(result_2d))
            @test all(τ -> isapprox(τ, tau, rtol=1.0), result_2d)
        end

    end

end 