using Test
using INT.ACW

@testset "ACW Module Tests" begin
    # Test data setup
    fs = 100.0  # 100 Hz sampling rate
    t = 0:1/fs:10  # 10 seconds of data
    # Create synthetic data with known properties
    # Using a damped oscillator with noise
    freq = 5.0  # 5 Hz oscillation
    tau = 0.5   # 0.5s decay time
    signal = exp.(-t/tau) .* sin.(2π*freq*t) 
    noise = 0.1 * randn(length(t))
    data = signal + noise

    @testset "Basic ACW Container" begin
        container = acw_container(data, fs, :acw0, nothing, nothing, [0.0])
        @test container.data == data
        @test container.fs == fs
        @test container.acwtypes == :acw0
    end

    @testset "ACW Types Validation" begin
        # Test single ACW type
        result_single = acw(data, fs, :acw0)
        @test length(result_single) == 1
        
        # Test multiple ACW types
        result_multiple = acw(data, fs, [:acw0, :acw50])
        @test length(result_multiple) == 2

        # Test invalid ACW type
        @test_throws ErrorException acw(data, fs, :invalid_type)
    end

    @testset "ACW Calculations" begin
        # Test ACW-0
        result = acw(data, fs, :acw0)
        @test length(result) == 1
        @test !isnothing(result[1])
        @test all(isfinite.(result[1]))

        # Test ACW-50
        result = acw(data, fs, :acw50)
        @test length(result) == 1
        @test !isnothing(result[1])
        @test all(isfinite.(result[1]))

        # Test tau calculation
        result = acw(data, fs, :tau)
        @test length(result) == 1
        @test !isnothing(result[1])
        @test all(isfinite.(result[1]))
        # Test if tau is roughly close to the input tau (with some tolerance)
        @test isapprox(result[1][1], tau, rtol=0.5)
    end

    @testset "Frequency Limits and N_lags" begin
        # Test with custom n_lags
        n_lags = 100
        result = acw(data, fs, :acw0, n_lags)
        @test length(result[1]) ≤ n_lags

        # Test with frequency limits
        freqlims = (0.1, 10.0)
        result = acw(data, fs, :knee, nothing, freqlims)
        @test length(result) == 1
        @test !isnothing(result[1])
    end

    @testset "Multi-dimensional Input" begin
        # Create 2D data
        data_2d = hcat(data, data)
        result = acw(data_2d, fs, :acw0)
        @test size(result[1], 2) == size(data_2d, 2)
    end
end 