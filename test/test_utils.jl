using Test
using BayesianINT.Utils
using Random
using BayesianINT.OrnsteinUhlenbeck
using BayesianINT.SummaryStats

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
        
        # Flat PSD
        flat_psd = ones(length(freqs))
        @test isnan(find_knee_frequency(flat_psd, freqs)[2])
        
        # Very noisy data
        random_psd = randn(length(freqs))
        @test isnan(find_knee_frequency(random_psd, freqs)[2])
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