using BayesianINT
using Test

@testset "BayesianINT.jl" begin
    # Include all test files
    include("test_summary_stats.jl")
    include("test_distances.jl")
    include("test_models.jl")
    include("test_abc.jl")
    include("test_ou_process.jl")
    include("test_ou_inference.jl")
end
