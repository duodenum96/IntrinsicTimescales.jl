# test/test_ou_process.jl
using Test
using Statistics
using Distributions
using BayesianINT

@testset "OU Process Generation" begin
    @testset "Basic properties" begin
        for backend in ["vanilla", "sciml"]
            @testset "$backend backend" begin
                tau = 0.5
                D = 2.0
                dt = 0.001
                T = 10.0
                num_trials = 100
                n_lags = 3000
                
                ou = generate_ou_process(tau, D, dt, T, num_trials, backend=backend)

                # Test dimensions
                @test size(ou) == (num_trials, Int(T/dt))
                
                # Test mean and variance
                @test mean(mean(ou, dims=2)) < 0.1  # Should be close to 0
                @test mean(abs.(std(ou, dims=2) .- D)) < 0.1  # Should be close to D
                
                # Test autocorrelation
                ac = mean([cor(@view(ou[i,1:end-1]), @view(ou[i,2:end])) for i in 1:num_trials])
                theoretical_ac = exp(-dt/tau)
                @test mean(abs.(ac .- theoretical_ac)) < 0.01
            end
        end
    end
    
    @testset "Edge cases" begin
        for backend in ["vanilla", "sciml"]
            @testset "$backend backend" begin
                # Test very small tau
                @test_nowarn generate_ou_process(0.1, 10.0, 0.01, 1.0, 10, backend=backend)
                
                # Test very large tau
                @test_nowarn generate_ou_process(1000.0, 0.001, 1.0, 100.0, 10, backend=backend)
                
                # Test single trial
                ou_single = generate_ou_process(20.0, 0.05, 1.0, 100.0, 1, backend=backend)
                @test size(ou_single, 1) == 1
            end
        end
    end

    @testset "Backend comparison" begin
        tau = 0.5
        D = 2.0
        dt = 0.001
        T = 10.0  # Reduced from 1000.0 for faster testing
        num_trials = 100
        
        # Generate data with both backends
        ou_vanilla = generate_ou_process(tau, D, dt, T, num_trials, backend="vanilla")
        ou_sciml = generate_ou_process(tau, D, dt, T, num_trials, backend="sciml")

        # Test dimensions match
        @test size(ou_vanilla) == size(ou_sciml)
        
        # Test statistical properties are similar between implementations
        @test abs(mean(ou_vanilla) - mean(ou_sciml)) < 0.1
        @test abs(std(ou_vanilla) - std(ou_sciml)) < 0.1
        
        # Test autocorrelations are similar
        ac_vanilla = mean([cor(@view(ou_vanilla[i,1:end-1]), @view(ou_vanilla[i,2:end])) for i in 1:num_trials])
        ac_sciml = mean([cor(@view(ou_sciml[i,1:end-1]), @view(ou_sciml[i,2:end])) for i in 1:num_trials])
        @test abs(ac_vanilla - ac_sciml) < 0.1
    end

    # Test invalid backend
    @test_throws ErrorException generate_ou_process(0.5, 2.0, 0.001, 10.0, 100, backend="invalid")
end