using IntrinsicTimescales
using Test

@testset "IntrinsicTimescales.jl" begin
    # Include all test files
    include("test_summary_stats.jl")
    include("test_distances.jl")
    include("test_ou_process.jl")
    include("test_acw.jl")
    include("test_utils.jl")
    include("test_abc.jl")
    include("test_abcdez.jl")
    include("test_one_timescale.jl")
    include("test_one_timescale_and_osc.jl")
    include("test_one_timescale_with_missing.jl")
    include("test_one_timescale_and_osc_with_missing.jl")
end
