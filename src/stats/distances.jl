# src/stats/distances.jl
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