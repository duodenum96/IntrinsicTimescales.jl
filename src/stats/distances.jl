# src/stats/distances.jl
module Distances
using Statistics
export linear_distance, logarithmic_distance
"""
Compute linear distance between summary statistics
"""
function linear_distance(data::Union{AbstractArray, Real}, synth_data::Union{AbstractArray, Real})
    return mean(abs2.(data .- synth_data))
end

"""
Compute logarithmic distance between summary statistics
"""
function logarithmic_distance(data::Union{AbstractArray, Real}, synth_data::Union{AbstractArray, Real})
    return mean(abs2.(log.(data) .- log.(synth_data)))
end
end # module