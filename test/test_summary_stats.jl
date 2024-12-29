# test/test_summary_stats.jl
using Statistics
using Test
using BayesianINT

@testset "Summary Statistics" begin
    @testset "Autocorrelation FFT" begin
        # Generate simple test signal
        t = 0:0.1:10
        signal = sin.(t)
        data = reshape(signal, 1, :)

        ac = comp_ac_fft(data; n_lags=100)

        # Test basic properties
        @test length(ac) == length(signal) - 1
        @test ac[1]≈1.0 atol=0.1  # First lag should be close to 1
        @test all(ac .<= 1.0)  # All values should be <= 1

        # Test periodicity detection
        period_samples = round(Int, 2π / 0.1)  # Number of samples in one period
        @test diff(ac)[period_samples]≈0.0 atol=0.1 # Local minimum
    end

    @testset "Power Spectral Density" begin
        # Generate test signal with known frequency
        fs = 100.0
        t = 0:1/fs:10
        f0 = 5.0  # 5 Hz signal
        signal = sin.(2π * f0 * t)

        psd, freq = comp_psd(signal, fs)
        freq_resolution = freq[2] - freq[1]

        # Find peak frequency
        peak_freq = freq[argmax(psd)]

        # Test if peak is at expected frequency
        @test abs(peak_freq - f0) < freq_resolution
    end

    @testset "Cross-correlation" begin
        # Test with identical signals (should give autocorrelation)
        signal = randn(1, 100)
        max_lag = 10
        cc = comp_cc(signal, signal, max_lag, 100)

        @test cc[1]≈var(signal) atol=0.1
        @test all(abs.(cc) .<= cc[1])  # Max at zero lag

        # Test with shifted signals
        shift = 5
        signal2 = hcat(zeros(1, shift), signal[:, 1:end-shift])
        cc_shifted = comp_cc(signal, signal2, max_lag, 100)
        @test argmax(cc_shifted) ≈ shift + 1

        # Should give similar result to fft-based autocorrelation
        t = 0:0.1:10
        signal = sin.(t)
        data = reshape(signal, 1, :)

        ac = comp_ac_fft(data; n_lags=length(t)-1)
        cc = comp_ac_time(data, length(t)-2)
        @test maximum(abs.(cc[1:50] - ac[1:50])) < 0.1 # This irks me. Isn't 0.05 too high?
    end
end