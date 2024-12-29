module BayesianINT

using Revise
using Reexport
using Statistics
using Distributions

# First include the core model definitions
include("core/model.jl")
@reexport using .Models

# Then include ABC and parallel implementations
include("core/abc.jl")
@reexport using .ABC

include("core/parallel.jl")
@reexport using .ParallelABC

# Include other modules
include("stats/summary.jl")
@reexport using .SummaryStats

include("stats/distances.jl")
@reexport using .Distances

include("models/ou_process.jl")
include("models/two_timescale.jl")
@reexport using .OrnsteinUhlenbeck
@reexport using .TwoTimescaleModels

export AbstractTimescaleModel,
       BaseModel,
       generate_ou_process,
       linear_distance,
       logarithmic_distance,
       OneTimescaleModel,
       TwoTimescaleModel

end # module