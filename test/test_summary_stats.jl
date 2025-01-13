# test/test_summary_stats.jl
using Statistics
using Test
using BayesianINT

# Helper function to find local maxima
function findlocalmaxima(x)
    indices = Int[]
    for i in 2:(length(x)-1)
        if x[i] > x[i-1] && x[i] > x[i+1]
            push!(indices, i)
        end
    end
    return indices
end 

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
        dt = 0.001
        T = 10.0
        τ = 1.0  # time constant
        D = 1.0  # noise amplitude
        t = dt:dt:T
        num_trials = 30
        ou_process = generate_ou_process(τ, D, dt, T, num_trials)
        data = ou_process    
        max_lags = 1200

        ac = comp_ac_fft(data; n_lags=max_lags)
        cc = comp_ac_time(data, max_lags)
        @test maximum(abs.(cc - ac)) < 0.05 # The difference is only for later lags close to zero. 
    end
end

@testset "Autocorrelation with missing data" begin
    # Test 1: Compare methods on complete data (OU process)
    dt = 0.01
    T = 10.0
    τ = 1.0  # time constant
    D = 1.0  # noise amplitude
    t = dt:dt:T
    num_trials = 20
    ou_process = generate_ou_process(τ, D, dt, T, num_trials)
    
    max_lags = 100
    ac_time = comp_ac_time(ou_process, max_lags)
    ac_time_missing = comp_ac_time_missing(ou_process, max_lags)
    ac_fft = comp_ac_fft(ou_process; n_lags=max_lags)
    
    # Test that all methods give similar results
    @test maximum(abs.(ac_time - ac_time_missing)) < 0.001 # error: 2e-16
    @test maximum(abs.(ac_time_missing - ac_fft)) < 0.05
    
    # Test theoretical decay of OU process
    lags = (0:(max_lags-1)) * dt
    theoretical_ac = exp.(-lags/τ)
    @test cor(ac_time_missing, theoretical_ac) > 0.95  # Strong correlation with theory
    
    # Test 2: Data with missing values
    ou_missing = copy(ou_process)
    
    # Insert missing values randomly (10% of data)
    n_missing = div(length(t), 10)
    missing_indices = rand(1:length(t), n_missing)
    ou_missing[1,missing_indices] .= NaN
    
    ac_missing = comp_ac_time_missing(ou_missing, max_lags)
    
    # Test that autocorrelation still decays
    @test ac_missing[1] ≈ 1.0 atol=0.1  # Should start near 1
    @test ac_missing[end] < ac_missing[1]  # Should decay
    @test issorted(ac_missing[1:div(end,2)], rev=true)  # First half should be monotonically decreasing

    # Test 3: Randomly insert one missing value to a data point and check if ACF is approximately the same
    ou_missing = copy(ou_process)
    missing_index = rand(1:length(t))
    ou_missing[1, missing_index] = NaN
    ac_missing_single = comp_ac_time_missing(ou_missing, max_lags)
    @test maximum(abs.(ac_missing_single - ac_time)) < 0.001
end

@testset "PSD Implementations" begin
    # Generate test data
    fs = 100.0  # 100 Hz sampling rate
    t = 0:1/fs:1  # 1 second of data
    f1, f2 = 10.0, 25.0  # Two frequency components
    
    # Create a signal with two sine waves
    signal = sin.(2π * f1 * t) .+ 0.5 * sin.(2π * f2 * t)
    # Add some noise
    noisy_signal = signal .+ 0.1 * randn(length(t))
    
    # Reshape to match expected input format (trials × samples)
    x = noisy_signal
    
    # Compute PSDs using both methods
    power_dsp, freq_dsp = comp_psd(x, fs, method="periodogram")
    power_ad, freq_ad = comp_psd_adfriendly(x[:], fs)
    
    # Check if peaks are at expected frequencies
    peak_indices_dsp = findlocalmaxima(power_dsp)
    peak_indices_ad = findlocalmaxima(power_ad)
    
    peak_freqs_dsp = freq_dsp[peak_indices_dsp]
    peak_freqs_ad = freq_ad[peak_indices_ad]
    
    # Check if we can detect both frequency components in both implementations
    @test any(isapprox.(peak_freqs_dsp, f1, rtol=1.0))
    @test any(isapprox.(peak_freqs_dsp, f2, rtol=1.0))
    @test any(isapprox.(peak_freqs_ad, f1, rtol=1.0))
    @test any(isapprox.(peak_freqs_ad, f2, rtol=1.0))

    # Plot to check
    # plot(freq_dsp, power_dsp)
    # plot!(freq_ad, power_ad)
    # vline!([f1])
    # vline!([f2])
end

@testset "Lomb-Scargle PSD" begin
    # Generate test data with known frequencies
    fs = 100.0  # 100 Hz sampling rate
    dt = 1/fs
    t = 0:dt:10  # 10 seconds of data
    f1, f2 = 5.0, 15.0  # Two frequency components
    
    # Create signal with two sine waves
    signal = sin.(2π * f1 * t) .+ 0.5 * sin.(2π * f2 * t)
    
    # Test 1: Single trial without missing data
    nanmask_single = fill(false, length(t))
    psd_single, freq_single = comp_psd_lombscargle(collect(t), signal, nanmask_single, dt)
    
    # Find peaks in the spectrum
    peak_indices = findlocalmaxima(psd_single)
    peak_freqs = freq_single[peak_indices]
    
    # Check if we can detect both frequency components
    @test any(isapprox.(peak_freqs, f1, rtol=0.1))
    @test any(isapprox.(peak_freqs, f2, rtol=0.1))
    
    # Test 2: Multiple trials
    n_trials = 5
    data = repeat(signal', n_trials, 1)  # Create matrix with identical trials
    nanmask_multi = fill(false, n_trials, length(t))
    
    psd_multi, freq_multi = comp_psd_lombscargle(collect(t), data, nanmask_multi, dt)
    
    # Find peaks in the multi-trial spectrum
    peak_indices_multi = findlocalmaxima(psd_multi)
    peak_freqs_multi = freq_multi[peak_indices_multi]
    
    # Check if we can detect both frequency components
    @test any(isapprox.(peak_freqs_multi, f1, rtol=0.1))
    @test any(isapprox.(peak_freqs_multi, f2, rtol=0.1))
    
    # Test 3: Data with missing values
    # Create random missing data pattern (10% missing)
    n_missing = div(length(t), 10)
    nanmask_missing = fill(false, n_trials, length(t))
    for trial in 1:n_trials
        missing_indices = rand(1:length(t), n_missing)
        nanmask_missing[trial, missing_indices] .= true
        data[trial, missing_indices] .= NaN
    end
    
    psd_missing, freq_missing = comp_psd_lombscargle(collect(t), data, nanmask_missing, dt)
    
    # Find peaks in the spectrum with missing data
    peak_indices_missing = findlocalmaxima(psd_missing)
    peak_freqs_missing = freq_missing[peak_indices_missing]
    
    # Check if we can still detect both frequency components
    @test any(isapprox.(peak_freqs_missing, f1, rtol=0.1))
    @test any(isapprox.(peak_freqs_missing, f2, rtol=0.1))
    
    # Test 4: Check prepare_lombscargle function
    times_masked, signal_masked, freq_grid = prepare_lombscargle(collect(t), data, nanmask_missing, dt)
    
    # Check dimensions
    @test length(times_masked) == n_trials
    @test length(signal_masked) == n_trials
    
    # Check that masked data excludes NaNs
    for trial in 1:n_trials
        @test !any(isnan.(signal_masked[trial]))
        @test length(times_masked[trial]) == length(signal_masked[trial])
        @test length(times_masked[trial]) == count(.!nanmask_missing[trial, :])
    end
    
    # Check frequency grid
    @test issorted(freq_grid)
    @test freq_grid[1] > 0  # Should start above 0
    @test freq_grid[end] <= fs/2  # Should not exceed Nyquist frequency
end

