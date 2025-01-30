"""
    ACW

Module providing autocorrelation width (ACW) calculations for time series analysis, including:
- ACW-0 (zero-crossing)
- ACW-50 (50% decay)
- ACW-euler (1/e decay)
- Exponential decay timescale (tau)
- Knee frequency estimation
"""
module ACW

using INT
using NaNStatistics

export acw, acw_container

"""
    acw_container

Structure holding ACW analysis inputs and results.

# Fields
- `data::AbstractArray{<:Real}`: Input time series data
- `fs::Real`: Sampling frequency
- `acwtypes::Union{Vector{<:Symbol}, Symbol, Nothing}`: Types of ACW to compute
- `n_lags::Union{Int, Nothing}`: Number of lags for ACF calculation
- `freqlims::Union{Tuple{Real, Real}, Nothing}`: Frequency limits for spectral analysis
- `acw_results::Vector{<:Real}`: Computed ACW values

# Notes
- Supported ACW types: :acw0, :acw50, :acweuler, :tau, :knee
- Results order matches input acwtypes order
"""
struct acw_container
    data::AbstractArray{<:Real}
    fs::Real
    acwtypes::Union{Vector{<:Symbol}, Symbol, Nothing} # Types of ACW: ACW-50, ACW-0, ACW-euler, tau, knee frequency
    n_lags::Union{Int, Nothing}
    freqlims::Union{Tuple{Real, Real}, Nothing}
    acw_results::Vector{<:Real}
end

possible_acwtypes = [:acw0, :acw50, :acweuler, :tau, :knee]

"""
    acw(data, fs; acwtypes=possible_acwtypes, n_lags=nothing, freqlims=nothing, dims=ndims(data))

Compute various autocorrelation width measures for time series data.

# Arguments
- `data::AbstractArray{<:Real}`: Input time series data
- `fs::Real`: Sampling frequency
- `acwtypes::Union{Vector{Symbol}, Symbol}=possible_acwtypes`: Types of ACW to compute
- `n_lags::Union{Int, Nothing}=nothing`: Number of lags for ACF calculation
- `freqlims::Union{Tuple{Real, Real}, Nothing}=nothing`: Frequency limits for spectral analysis
- `dims::Int=ndims(data)`: Dimension along which to compute ACW

# Returns
- Vector of computed ACW measures, ordered according to input acwtypes

# Notes
- Supported ACW types:
  * :acw0 - Time to first zero crossing
  * :acw50 - Time to 50% decay
  * :acweuler - Time to 1/e decay
  * :tau - Exponential decay timescale
  * :knee - Knee frequency from spectral analysis
- If n_lags is not specified, uses 1.1 * ACW0
- For spectral measures, freqlims defaults to full frequency range
"""
function acw(data, fs; acwtypes=possible_acwtypes, n_lags=nothing, freqlims=nothing,
             dims=ndims(data))

    if data isa AbstractVector
        data = reshape(data, (1, length(data)))
        dims = ndims(data)
    end

    if acwtypes isa Symbol
        acwtypes = [acwtypes]
    end

    dt = 1.0 / fs
    acf_acwtypes = [:acw0, :acw50, :acweuler, :tau]
    n_acw = length(acwtypes)
    if n_acw == 0
        error("No ACW types specified. Possible ACW types: $(possible_acwtypes)")
    end
    # Check if any requested acwtype is not in possible_acwtypes
    result = Vector{AbstractArray{<:Real}}(undef, n_acw)
    acwtypes = check_acwtypes(acwtypes, possible_acwtypes)

    if any(in.(acf_acwtypes, [acwtypes]))
        acf = comp_ac_fft(data; dims=dims)
        lags_samples = 0.0:(size(data, dims)-1)
        lags = lags_samples * dt

        acw0_sample = acw0(lags_samples, acf; dims=dims)
        if any(in.(:acw0, [acwtypes]))
            acw0_idx = findfirst(acwtypes .== :acw0)
            acw0_result = acw0(lags, acf; dims=dims)
            result[acw0_idx] = acw0_result
        end

        if any(in.(:acw50, [acwtypes]))
            acw50_idx = findfirst(acwtypes .== :acw50)
            acw50_result = acw50(lags, acf; dims=dims)
            result[acw50_idx] = acw50_result
        end
        if any(in.(:acweuler, [acwtypes]))
            acweuler_idx = findfirst(acwtypes .== :acweuler)
            acweuler_result = acweuler(lags, acf; dims=dims)
            result[acweuler_idx] = acweuler_result
        end

        if isnothing(n_lags)
            n_lags = 1.1 * nanmaximum(acw0_sample)
            if isnan(n_lags)
                n_lags = size(data, dims)
            else
                n_lags = ceil(Int, n_lags)
            end
        end
        acf = selectdim(acf, dims, 1:n_lags)
        lags = lags[1:n_lags]

        if any(in.(:tau, [acwtypes]))
            tau_idx = findfirst(acwtypes .== :tau)
            tau_result = fit_expdecay(collect(lags), acf; dims=dims)
            result[tau_idx] = tau_result
        end
    end

    if any(in.(:knee, [acwtypes]))
        knee_idx = findfirst(acwtypes .== :knee)
        fs = 1 / dt
        psd, freqs = comp_psd(data, fs, dims=dims)
        if isnothing(freqlims)
            freqlims = (freqs[1], freqs[end])
        end
        knee_result = tau_from_knee(find_knee_frequency(psd, freqs; dims=dims, min_freq=freqlims[1], max_freq=freqlims[2]))
        result[knee_idx] = knee_result
    end
    return result
end

end
