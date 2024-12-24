# test/test_summary_stats.jl
@testset "Summary Statistics" begin
    @testset "Autocorrelation FFT" begin
        # Generate simple test signal
        t = 0:0.1:10
        signal = sin.(t)
        data = reshape(signal, 1, :)
        
        ac = comp_ac_fft(data)
        
        # Test basic properties
        @test length(ac) == length(signal) - 1
        @test ac[1] ≈ 1.0 atol=0.1  # First lag should be close to 1
        @test all(ac .<= 1.0)  # All values should be <= 1
        
        # Test periodicity detection
        period_samples = round(Int, 2π/0.1)  # Number of samples in one period
        @test ac[period_samples] ≈ 1.0 atol=0.1
    end
    
    @testset "Power Spectral Density" begin
        # Generate test signal with known frequency
        fs = 100.0
        t = 0:1/fs:10
        f0 = 5.0  # 5 Hz signal
        signal = sin.(2π * f0 * t)
        data = reshape(signal, 1, :)
        
        psd = comp_psd(data, 10.0, 1/fs)
        
        # Find peak frequency
        freq_resolution = fs/(2*length(t))
        peak_idx = argmax(psd)
        peak_freq = (peak_idx-1) * freq_resolution
        
        # Test if peak is at expected frequency
        @test abs(peak_freq - f0) < freq_resolution
    end
    
    @testset "Cross-correlation" begin
        # Test with identical signals (should give autocorrelation)
        signal = randn(1, 100)
        cc = comp_cc(signal, signal, 10, 1.0, 100)
        
        @test cc[1] ≈ var(signal) atol=0.1
        @test all(abs.(cc) .<= cc[1])  # Max at zero lag
        
        # Test with shifted signals
        shift = 5
        signal2 = hcat(zeros(1, shift), signal[:, 1:end-shift])
        cc_shifted = comp_cc(signal, signal2, 10, 1.0, 100)
        @test argmax(cc_shifted) ≈ shift + 1
    end
end