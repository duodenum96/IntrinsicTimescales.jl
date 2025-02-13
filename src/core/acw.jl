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

using Revise
using IntrinsicTimescales
using NaNStatistics, Statistics

export acw, ACWResults

"""
    ACWResults

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
struct ACWResults
    fs::Real
    acw_results
    acwtypes::Union{Vector{<:Symbol}, Symbol} # Types of ACW: ACW-50, ACW-0, ACW-euler, tau, knee frequency
    n_lags::Union{Int, Nothing}
    freqlims::Union{Tuple{Real, Real}, Nothing}
    acf::Union{AbstractVector{<:Real}, AbstractArray{<:Real}, Nothing}
    psd::Union{AbstractVector{<:Real}, AbstractArray{<:Real}, Nothing}
    freqs::Union{AbstractVector{<:Real}, AbstractArray{<:Real}, Nothing}
    lags::Union{AbstractVector{<:Real}, AbstractArray{<:Real}, Nothing}
    x_dim::Union{Int, Nothing}
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
- `time::Union{Vector{Real}, Nothing}=nothing`: Time vector. This is required for Lomb-Scargle method in the case of missing data.
- `dims::Int=ndims(data)`: Dimension along which to compute ACW (Dimension of time)
- `return_acf::Bool=true`: Whether to return the ACF
- `return_psd::Bool=true`: Whether to return the PSD
- `average_over_trials::Bool=false`: Whether to average the ACF or PSD over trials
- `trial_dims::Int=1`: Dimension along which to average the ACF or PSD over trials (Dimension of trials)

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
function acw(data, fs; acwtypes=possible_acwtypes, n_lags=nothing, freqlims=nothing, time=nothing, 
             dims=ndims(data), return_acf=true, return_psd=true, average_over_trials=false,
             trial_dims=1)

    missingmask = ismissing.(data)
    if any(missingmask)
        data[missingmask] .= NaN
    end

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
    result = Vector(undef, n_acw)
    acwtypes = check_acwtypes(acwtypes, possible_acwtypes)

    nanmask = isnan.(data)
    iscomplete = !any(nanmask)

    if any(in.(acf_acwtypes, [acwtypes]))
        if iscomplete
            acf = comp_ac_fft(data; dims=dims)
        else
            if isnothing(n_lags)
                acf = comp_ac_time_missing(data; dims=dims)
            else
                acf = comp_ac_time_missing(data; dims=dims, n_lags=n_lags)
            end
        end

        if average_over_trials
            acf = mean(acf, dims=trial_dims)
        end

        lags_samples = 0.0:(size(data, dims)-1)
        lags = lags_samples * dt

        acw0_sample = acw0(lags_samples, acf; dims=dims)
        if any(in.(:acw0, [acwtypes]))
            acw0_idx = findfirst(acwtypes .== :acw0)
            acw0_result = acw0(lags, acf; dims=dims)
            if (acw0_result isa Vector) && (length(acw0_result) == 1)
                result[acw0_idx] = acw0_result[1]
            else
                result[acw0_idx] = acw0_result
            end
        end

        if any(in.(:acw50, [acwtypes]))
            acw50_idx = findfirst(acwtypes .== :acw50)
            acw50_result = acw50(lags, acf; dims=dims)
            if (acw50_result isa Vector) && (length(acw50_result) == 1)
                result[acw50_idx] = acw50_result[1]
            else
                result[acw50_idx] = acw50_result
            end
        end
        if any(in.(:acweuler, [acwtypes]))
            acweuler_idx = findfirst(acwtypes .== :acweuler)
            acweuler_result = acweuler(lags, acf; dims=dims)
            if (acweuler_result isa Vector) && (length(acweuler_result) == 1)
                result[acweuler_idx] = acweuler_result[1]
            else
                result[acweuler_idx] = acweuler_result
            end
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
            if (tau_result isa Vector) && (length(tau_result) == 1)
                result[tau_idx] = tau_result[1]
            else
                result[tau_idx] = tau_result
            end
        end

        if data isa Vector
            acf = acf[:]
        end
    else
        lags = nothing
        acf = nothing
        n_lags = nothing
    end

    if any(in.(:knee, [acwtypes]))
        knee_idx = findfirst(acwtypes .== :knee)
        if iscomplete
            psd, freqs = comp_psd(data, fs, dims=dims)
        else
            if isnothing(time)
                raise(ArgumentError("Time vector is required for Lomb-Scargle method in the case of missing data.\n" * 
                        "Call the function as `acw(data, fs; time=time)`"))
            end
            psd, freqs = comp_psd_lombscargle(time, data, nanmask, dt; dims=dims)
        end

        if average_over_trials
            psd = mean(psd, dims=trial_dims)
        end

        if isnothing(freqlims)
            freqlims = (freqs[1], freqs[end])
        end
        knee_result = tau_from_knee(find_knee_frequency(psd, freqs; dims=dims, min_freq=freqlims[1], max_freq=freqlims[2]))
        if (knee_result isa Vector) && (length(knee_result) == 1)
            result[knee_idx] = knee_result[1]
        else
            result[knee_idx] = knee_result
        end

        if data isa Vector
            psd = psd[:]
        end

    else
        freqlims = nothing
        psd = nothing
        freqs = nothing
    end

    if !return_acf
        acf = nothing
        lags = nothing
    end
    if !return_psd
        psd = nothing
        freqs = nothing
    end

    # Find the dimension of x axis (lags or freqs)
    if !isnothing(lags)
        x_dim = findfirst(size(lags) .== size(acf))
    elseif !isnothing(freqs)
        x_dim = findfirst(size(freqs) .== size(psd))
    else
        x_dim = nothing
    end

    if n_acw == 1
        return ACWResults(fs, result[1], acwtypes, n_lags, freqlims, acf, psd, freqs, lags, x_dim)
    else
        return ACWResults(fs, result, acwtypes, n_lags, freqlims, acf, psd, freqs, lags, x_dim)
    end
end


end
