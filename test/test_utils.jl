using Test
using IntrinsicTimescales.Utils
using Random
using IntrinsicTimescales.OrnsteinUhlenbeck
using IntrinsicTimescales.SummaryStats

@testset "find_knee_frequency tests" begin
    # Test 1: Perfect Lorentzian
    @testset "Ideal Lorentzian" begin
        freqs = collect(0.0:0.1:100.0) / 1000.0
        f_knee = 10.0 / 1000.0
        amp = 100.0
        psd = @. amp / (1 + (freqs/f_knee)^2)
        
        detected_knee = find_knee_frequency(psd, freqs)[2]
        @test isapprox(detected_knee, f_knee, rtol=0.1)
    end
    
    # Test 2: Noisy Lorentzian
    @testset "Noisy Lorentzian" begin
        Random.seed!(123)
        freqs = collect(0.0:0.1:100.0) / 1000.0
        f_knee = 10.0 / 1000.0
        amp = 100.0
        base_psd = @. amp / (1 + (freqs/f_knee)^2)
        noise = randn(length(freqs)) * 0.05 * amp
        noisy_psd = base_psd + noise
        
        detected_knee = find_knee_frequency(noisy_psd, freqs)[2]
        @test isapprox(detected_knee, f_knee, rtol=0.2)
    end
    
    # Test 3: Different frequency ranges
    @testset "Different frequency ranges" begin
        freqs = collect(0.0:0.1:100.0) / 1000.0
        f_knee = 10.0 / 1000.0
        amp = 100.0
        psd = @. amp / (1 + (freqs/f_knee)^2)
        
        knee1 = find_knee_frequency(psd, freqs, min_freq=0.1/1000.0, max_freq=50.0)[2]
        knee2 = find_knee_frequency(psd, freqs, min_freq=1.0 / 1000.0, max_freq=20.0 / 1000.0)[2]
        
        @test isapprox(knee1, knee2, rtol=0.2)
        @test isapprox(knee1, f_knee, rtol=0.2)
    end
    
    # Test 4: Edge cases
    @testset "Edge cases" begin
        freqs = collect(0.0:0.1:100.0) / 1000.0

        # TODO: When the knee frequency is not detected, the
        # algorithm just returns the maximum frequency. We should warn the user about this.
        
        # Flat PSD
        flat_psd = ones(length(freqs))
        @test isapprox(find_knee_frequency(flat_psd, freqs)[2], freqs[end])
        
        # Very noisy data
        random_psd = randn(length(freqs))
        @test isapprox(find_knee_frequency(random_psd, freqs)[2], freqs[end], rtol=0.01)
    end
    
    # Test 5: OU process
    @testset "OU process" begin
        # No oscillation
        tau = 10.0
        dt = 1.0
        T = 5000.0
        num_trials = 100
        ou = generate_ou_process(tau, 1.0, dt, T, num_trials)
        psd, freqs = comp_psd(ou, 1/dt)
        detected_knee = find_knee_frequency(psd, freqs, min_freq = freqs[1], max_freq = freqs[end])[2]
        # Theoretically, tau = 1 / 2pi * f_knee
        @test isapprox(tau, (1 / (2π * detected_knee)), rtol=2)

        # Oscillation
        true_tau = 100.0
        true_freq = 10.0 / 1000.0  # mHz
        true_coeff = 0.95  # oscillation coefficient
        dt = 1.0
        T = 30000.0
        num_trials = 10
        
        data = generate_ou_with_oscillation([true_tau, true_freq, true_coeff],
                                            dt, T, num_trials, 0.0, 1.0)
        psd, freqs = comp_psd(data, 1/dt)
        detected_knee = find_knee_frequency(psd, freqs, min_freq = freqs[1], max_freq = freqs[end])[2]
        # Theoretically, tau = 1 / 2pi * f_knee
        @test isapprox(true_tau, (1 / (2π * detected_knee)), rtol=2)
    end
    
end

@testset "Exponential decay fitting tests" begin
    @testset "Basic exponential decay" begin
        # Create synthetic data with known tau
        true_tau = 10.0
        lags = collect(0.0:0.5:20.0)
        perfect_acf = exp.(-(1/true_tau) * lags)
        
        # Test expdecay function
        @test all(isapprox.(expdecay(true_tau, lags), perfect_acf))
        
        # Test fitting function
        fitted_tau = fit_expdecay(lags, perfect_acf)
        @test isapprox(fitted_tau, true_tau, rtol=0.01)
    end
    
    @testset "Noisy exponential decay" begin
        # Create synthetic data with noise
        Random.seed!(666)
        true_tau = 5.0
        lags = collect(0.0:0.5:20.0)
        perfect_acf = exp.(-(1/true_tau) * lags)
        noisy_acf = perfect_acf + randn(length(lags)) * 0.05
        
        # Test fitting with noisy data
        fitted_tau = fit_expdecay(lags, noisy_acf)
        @test isapprox(fitted_tau, true_tau, rtol=0.1)
    end
    
    @testset "Edge cases" begin
        lags = collect(0.0:0.5:200.0)
        
        # Test very small tau
        small_tau = 0.1
        small_acf = exp.(-(1/small_tau) * lags)
        fitted_small = fit_expdecay(lags, small_acf)
        @test isapprox(fitted_small, small_tau, rtol=0.1)
        
        # Test large tau
        large_tau = 100.0
        large_acf = exp.(-(1/large_tau) * lags)
        fitted_large = fit_expdecay(lags, large_acf)
        @test isapprox(fitted_large, large_tau, rtol=0.1)
    end
end

@testset "N-dimensional array handling" begin
    @testset "acw50 with different dimensions" begin
        # Test data setup
        lags = collect(0.0:0.5:20.0)
        tau = 5.0
        perfect_acf = exp.(-(1/tau) * lags)
        
        # 1D case (vector)
        acw50_1d = acw50(lags, perfect_acf)
        @test isapprox(acw50_1d, -tau * log(0.5), rtol=0.1)
        
        # 2D case (matrix)
        acf_2d = repeat(perfect_acf', 3, 1)  # 3 identical rows
        acw50_2d_cols = acw50(lags, acf_2d, dims=2)
        @test length(acw50_2d_cols) == 3
        @test all(x -> isapprox(x, -tau * log(0.5), rtol=0.1), acw50_2d_cols)
        
        # 2D case along rows
        acf_2d_t = acf_2d'  # transpose to test along rows
        acw50_2d_rows = acw50(lags, acf_2d_t, dims=1)
        @test length(acw50_2d_rows) == size(acf_2d_t, 2)
        @test all(x -> isapprox(x, -tau * log(0.5), rtol=0.1), acw50_2d_rows)
        
        # 3D case
        acf_3d = repeat(perfect_acf, 1, 3, 2)  # 3×2 identical slices
        acw50_3d = acw50(lags, acf_3d, dims=1)
        @test size(acw50_3d) == (3, 2)
        @test all(x -> isapprox(x, -tau * log(0.5), rtol=0.1), acw50_3d)
    end

    @testset "fit_expdecay with different dimensions" begin
        lags = collect(0.0:0.5:20.0)
        true_tau = 5.0
        perfect_acf = exp.(-(1/true_tau) * lags)
        
        # 1D case
        fitted_tau_1d = fit_expdecay(lags, perfect_acf)
        @test isapprox(fitted_tau_1d, true_tau, rtol=0.1)
        
        # 2D case - multiple trials
        acf_2d = repeat(perfect_acf', 3, 1)  # 3 trials
        fitted_taus_2d = fit_expdecay(lags, acf_2d, dims=2)
        @test length(fitted_taus_2d) == 3
        @test all(x -> isapprox(x, true_tau, rtol=0.1), fitted_taus_2d)
        
        # 3D case - multiple experiments
        acf_3d = repeat(perfect_acf, 1, 3, 2)  # 3×2 experiments
        fitted_taus_3d = fit_expdecay(lags, acf_3d, dims=1)
        @test size(fitted_taus_3d) == (3, 2)
        @test all(x -> isapprox(x, true_tau, rtol=0.1), fitted_taus_3d)
    end

    @testset "find_knee_frequency with different dimensions" begin
        freqs = collect(0.0:0.1:100.0) / 1000.0
        f_knee = 10.0 / 1000.0
        amp = 100.0
        perfect_psd = lorentzian(freqs, [amp, f_knee])
        
        # 1D case
        detected_knee_1d = find_knee_frequency(perfect_psd, freqs)[2]
        @test isapprox(detected_knee_1d, f_knee, rtol=0.1)
        
        # 2D case - multiple trials
        psd_2d = repeat(perfect_psd', 3, 1)  # 3 trials
        detected_knees_2d = find_knee_frequency(psd_2d, freqs, dims=2)
        @test length(detected_knees_2d) == 3
        @test all(x -> isapprox(x, f_knee, rtol=0.1), detected_knees_2d)
        
        # 3D case - multiple experiments
        psd_3d = repeat(perfect_psd, 1, 3, 2)  # 3×2 experiments
        detected_knees_3d = find_knee_frequency(psd_3d, freqs, dims=1)
        @test size(detected_knees_3d) == (3, 2)
        @test all(x -> isapprox(x, f_knee, rtol=0.1), detected_knees_3d)
    end

    @testset "fooof_fit with different dimensions" begin
        freqs = collect(0.0:0.1:100.0) / 1000.0
        f_knee = 5.0 / 1000.0
        f_peak = 30.0 / 1000.0
        amp = 100.0
        
        # Create PSD with both knee and peak
        base_psd = lorentzian(freqs, [amp, f_knee])
        peak_psd = @. 50 * exp(-(((freqs - f_peak)/(2/1000))^2))
        perfect_psd = base_psd + peak_psd
        
        # 1D case
        result_1d = fooof_fit(base_psd, freqs)
        @test isapprox(result_1d[1], f_knee, rtol=0.2)
        result_1d_osc = fooof_fit(perfect_psd, freqs, oscillation_peak=true)
        @test isapprox(result_1d_osc[2], f_peak, rtol=0.2)
        
        # 2D case - multiple trials
        psd_2d = repeat(base_psd', 3, 1)
        psd_2d_osc = repeat(perfect_psd', 3, 1)
        results_2d = fooof_fit(psd_2d, freqs, dims=2)
        results_2d_osc = fooof_fit(psd_2d_osc, freqs, dims=2, oscillation_peak=true)
        @test length(results_2d) == 3
        @test all(x -> isapprox(x[1], f_knee, rtol=0.2), results_2d)
        @test all(x -> isapprox(x[2], f_peak, rtol=0.2), results_2d_osc)
        
        # 3D case - multiple experiments
        psd_3d = repeat(base_psd, 1, 3, 2)
        psd_3d_osc = repeat(perfect_psd, 1, 3, 2)
        results_3d = fooof_fit(psd_3d, freqs, dims=1)
        results_3d_osc = fooof_fit(psd_3d_osc, freqs, dims=1, oscillation_peak=true)
        @test size(results_3d) == (3, 2)
        @test all(x -> isapprox(x[1], f_knee, rtol=0.2), results_3d)
        @test all(x -> isapprox(x[2], f_peak, rtol=0.2), results_3d_osc)
    end
end

