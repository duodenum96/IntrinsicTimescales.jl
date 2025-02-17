# test/test_ou_process.jl
using Test
using Statistics
using Distributions
using IntrinsicTimescales

@testset "OU Process Generation" begin
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

