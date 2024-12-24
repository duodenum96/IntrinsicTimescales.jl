module Models

export AbstractTimescaleModel, BaseModel

abstract type AbstractTimescaleModel end

"""
Base model interface for ABC computations
"""
struct BaseModel{T,D,P} <: AbstractTimescaleModel
    data::D
    prior::P
    data_sum_stats::T
    epsilon::Float64
end

# Required methods that need to be implemented for each model
function draw_theta end
function generate_data end
function summary_stats end
function distance_function end

# Combined generation and reduction step
function generate_data_and_reduce(model::AbstractTimescaleModel, theta)
    synth = generate_data(model, theta)
    sum_stats = summary_stats(model, synth)
    d = distance_function(model, sum_stats, model.data_sum_stats)
    return d
end

end # module
