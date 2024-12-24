# src/core/parallel.jl

"""
Parallel ABC implementation
"""
module ParallelABC

using Distributed
using ..Models  # Use relative module path
using ..ABC    # Assuming we have ABC module

export parallel_basic_abc, combine_results

"""
Run ABC algorithm in parallel
"""
function parallel_basic_abc(
    model::Models.AbstractTimescaleModel,
    n_procs::Int;
    samples_per_proc::Int,
    epsilon::Float64,
    max_iter::Int
)
    # Initialize worker pool
    worker_pool = CachingPool(workers()[1:n_procs])
    
    # Run ABC on each worker
    results = @distributed (combine_results) for i in 1:n_procs
        ABC.basic_abc(
            model;
            min_samples=samples_per_proc,
            epsilon=epsilon,
            max_iter=max_iter
        )
    end
    
    return results
end

"""
Combine results from parallel runs
"""
function combine_results(r1, r2)
    if r1 === nothing
        return r2
    elseif r2 === nothing
        return r1
    end
    
    samples1, dist1 = r1
    samples2, dist2 = r2
    
    return (
        vcat(samples1, samples2),
        vcat(dist1, dist2)
    )
end

end # module