# src/stats/distances.jl
module Distances
using Statistics
export linear_distance, logarithmic_distance
"""
Compute linear distance between autocorrelations
"""
function linear_distance(data::AbstractArray, synth_data::AbstractArray)
    return mean(abs2.(data .- synth_data))
end

"""
Compute logarithmic distance between autocorrelations
"""
function logarithmic_distance(data::AbstractArray, synth_data::AbstractArray)
    return mean(abs2.(log.(data) .- log.(synth_data)))
end
end # module