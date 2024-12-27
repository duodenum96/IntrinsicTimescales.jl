# test/test_ou_process.jl
using Statistics
@testset "OU Process Generation" begin
    @testset "Basic properties" begin
        tau = 20.0
        D = 1.0/tau
        deltaT = 1.0
        T = 1000.0
        num_trials = 100
        
        ou = generate_ou_process(tau, D, deltaT, T, num_trials)
        
        # Test dimensions
        @test size(ou) == (num_trials, Int(T/deltaT))
        
        # Test mean and variance
        @test abs(mean(ou)) < 0.1  # Should be close to 0
        @test abs(var(ou) - 1.0) < 0.1  # Should be close to 1
        
        # Test autocorrelation
        ac = mean([cor(@view(ou[i,1:end-1]), @view(ou[i,2:end])) for i in 1:num_trials])
        theoretical_ac = exp(-deltaT/tau)
        @test abs(ac - theoretical_ac) < 0.1
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
end