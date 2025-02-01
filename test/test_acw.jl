using Test
using INT.ACW
using INT
using Random

@testset "ACW Module Tests" begin
    # Test data setup
    fs = 100.0  # 100 Hz sampling rate
    t = 0:1/fs:10  # 10 seconds of data
    # Create synthetic data with known properties
    # Using a damped oscillator with noise
    freq = 5.0  # 5 Hz oscillation
    tau = 0.5   # 0.5s decay time
    signal = exp.(-t/tau)
    noise = 0.1 * randn(length(t))
    data = signal + noise

    @testset "Basic ACW Container" begin
        container = acw_container(fs, [0.0], :acw0, nothing, nothing, nothing, nothing, nothing, nothing)
        @test container.fs == fs
        @test container.acw_results == [0.0]
        @test container.acwtypes == :acw0
        @test container.n_lags === nothing
        @test container.freqlims === nothing
        @test container.acf === nothing
        @test container.psd === nothing
        @test container.freqs === nothing
        @test container.lags === nothing
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
        result = acw(data, fs, acwtypes=:acw0, n_lags=n_lags).acw_results
        @test length(result[1]) ≤ n_lags

        # Test with frequency limits
        freqlims = (0.1, 10.0)
        result = acw(data, fs, acwtypes=:knee, freqlims=freqlims).acw_results
        @test length(result) == 1
        @test !isnothing(result[1])
    end

    @testset "Multi-dimensional Input" begin
        # Create 2D data
        data_2d = hcat(data, data)
        result = acw(data_2d, fs, acwtypes=:acw0, dims=1).acw_results
        @test length(result[1]) == size(data_2d, 2)

        # Create 3D data
        data_3d = cat(data_2d, data_2d, dims=3)
        result = acw(data_3d, fs, acwtypes=:acw0, dims=1).acw_results
        @test size(result[1]) == (size(data_3d, 2), size(data_3d, 3))


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
        @test size(result_test[1]) == (3, 1) # Check output shape
        
        # Test if ACW50 values increase with increasing tau
        # ACW50 should be proportional to tau: acw50 ≈ -tau * log(0.5)
        @test result_test[1][1] < result_test[1][2] < result_test[1][3]
        @test isapprox(result_test[1][1], -tau1 * log(0.5), rtol=0.5)
        @test isapprox(result_test[1][2], -tau2 * log(0.5), rtol=0.5)
        @test isapprox(result_test[1][3], -tau3 * log(0.5), rtol=0.5)

        # Test tau estimation directly
        result_tau = acw(data_test, fs, acwtypes=:tau, dims=2).acw_results
        @test size(result_tau[1]) == (3, 1)
        @test isapprox(result_tau[1][1], tau1, rtol=1.0)
        @test isapprox(result_tau[1][2], tau2, rtol=0.5)
        @test isapprox(result_tau[1][3], tau3, rtol=0.5)

        # Test along dimension 2
        data_test_perm = permutedims(data_test, (2, 1, 3))
        result_test_dim2 = acw(data_test_perm, fs, acwtypes=:tau, dims=1).acw_results
        @test size(result_test_dim2[1]) == (3, 1)
        @test isapprox(result_test_dim2[1][1], tau1, rtol=1.0)
        @test isapprox(result_test_dim2[1][2], tau2, rtol=0.5)
        @test isapprox(result_test_dim2[1][3], tau3, rtol=0.5)
    end
end 