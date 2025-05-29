# src/stats/distances.jl
"""
    Distances

Module providing distance metrics for comparing summary statistics in ABC inference.
Currently implements linear (L2) and logarithmic distances.
"""
module Distances
using Statistics
export linear_distance, logarithmic_distance

"""
    linear_distance(data, synth_data)

Compute mean squared error (MSE) between summary statistics.

# Arguments
- `data::Union{AbstractArray, Real}`: Observed data summary statistics
- `synth_data::Union{AbstractArray, Real}`: Simulated data summary statistics

# Returns
- `Float64`: Mean squared difference between data and synth_data

# Notes
- Handles both scalar and array inputs
- For arrays, computes element-wise differences before averaging
- Useful for comparing summary statistics on linear scales
"""
function linear_distance(data::Union{AbstractArray, Real}, synth_data::Union{AbstractArray, Real})
    return mean(abs2.(data .- synth_data))
end

"""
    logarithmic_distance(data, synth_data)

Compute mean squared distance between logarithms of summary statistics.

# Arguments
- `data::Union{AbstractArray, Real}`: Observed data summary statistics
- `synth_data::Union{AbstractArray, Real}`: Simulated data summary statistics

# Returns
- `Float64`: Mean squared difference between log(data) and log(synth_data)

# Notes
- Handles both scalar and array inputs
- For arrays, computes element-wise log differences before averaging
- Useful for comparing summary statistics spanning multiple orders of magnitude
- Assumes all values are positive
"""
function logarithmic_distance(data::Union{AbstractArray, Real}, synth_data::Union{AbstractArray, Real})
    return mean(abs2.(log.(data) .- log.(synth_data)))
end
end # module