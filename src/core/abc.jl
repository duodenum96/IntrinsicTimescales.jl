# src/core/abc.jl
"""
Basic ABC rejection sampling algorithm
"""
function basic_abc(
    model::AbstractTimescaleModel;
    min_samples::Int,
    epsilon::Float64,
    max_iter::Int
)
    accepted_samples = []
    distances = Float64[]
    
    for i in 1:max_iter
        theta = draw_theta(model)
        d = generate_data_and_reduce(model, theta)
        
        if d <= epsilon
            push!(accepted_samples, theta)
            push!(distances, d)
        end
        
        length(accepted_samples) >= min_samples && break
    end
    
    return accepted_samples, distances
end