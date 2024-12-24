# src/models/ou_process.jl
using Random, Distributions

"""
Generate an OU process with a single timescale
"""
function generate_ou_process(
    tau::Float64,
    D::Float64,
    deltaT::Float64,
    T::Float64,
    num_trials::Int
)
    num_bin = Int(T / deltaT)
    noise = randn(num_trials, num_bin)
    ou = zeros(num_trials, num_bin)
    ou[:, 1] = noise[:, 1]

    for i in 2:num_bin
        ou[:, i] = @views ou[:, i-1] .- (ou[:, i-1] / tau) * deltaT .+
                          sqrt(2 * D * deltaT) * noise[:, i-1]
    end

    return ou
end