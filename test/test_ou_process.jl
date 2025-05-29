# test/test_ou_process.jl
using Test
using Statistics
using Distributions
using IntrinsicTimescales
using Random

@testset "OU Process tests" begin
    @testset "OU Process Generation" begin
        Random.seed!(123)
        tau = 0.5
        D = 2.0
        dt = 0.001
        T = 10.0
        num_trials = 10
        n_lags = 3000

        ou = generate_ou_process(tau, D, dt, T, num_trials)

        # Test dimensions
        @test size(ou) == (num_trials, Int(T / dt))

        # Test mean and variance
        @test mean(mean(ou, dims=2)) < 0.1  # Should be close to 0
        @test mean(abs.(std(ou, dims=2) .- D)) < 0.1  # Should be close to D

        # Test autocorrelation
        ac = mean([cor(@view(ou[i, 1:end-1]), @view(ou[i, 2:end]))
                    for i in 1:num_trials])
        theoretical_ac = exp(-dt / tau)
        @test mean(abs.(ac .- theoretical_ac)) < 0.01
    end

    @testset "Edge cases" begin
        # Test very small tau
        @test_nowarn generate_ou_process(0.1, 10.0, 0.01, 1.0, 10)

        # Test very large tau
        @test_nowarn generate_ou_process(1000.0, 0.001, 1.0, 100.0, 10)

        # Test single trial
        ou_single = generate_ou_process(20.0, 0.05, 1.0, 100.0, 1)
        @test size(ou_single, 1) == 1
    end

    @testset "Seed functionality" begin
        tau = 1.0
        D = 1.0
        dt = 0.01
        T = 5.0
        num_trials = 5
        seed = 42
        rng = Xoshiro(seed)

        # Generate two OU processes with the same seed
        ou1 = generate_ou_process_sciml(tau, D, dt, T, num_trials, true; rng=rng, deq_seed=seed)[1]
        rng = Xoshiro(seed)
        ou2 = generate_ou_process_sciml(tau, D, dt, T, num_trials, true; rng=rng, deq_seed=seed)[1]

        # They should be identical
        @test ou1 ≈ ou2

        # Generate with different seed
        ou3 = generate_ou_process_sciml(tau, D, dt, T, num_trials, true; rng=rng, deq_seed=seed+1)[1]

        # Should be different from the first two
        @test !(ou1 ≈ ou3)

        # Test with no seed (should work but be different each time)
        ou4 = generate_ou_process_sciml(tau, D, dt, T, num_trials, true)[1]
        ou5 = generate_ou_process_sciml(tau, D, dt, T, num_trials, true)[1]
        
        # These should likely be different (very small chance they're the same)
        @test !(ou4 ≈ ou5)
    end
end

