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

using IntrinsicTimescales
using NaNStatistics, Statistics
export acw, ACWResults

"""
    ACWResults

Structure holding ACW analysis inputs and results.

# Fields
- `fs::Real`: Sampling frequency
- `acw_results`: Computed ACW values (type depends on number of ACW types requested)
- `acwtypes::Union{Vector{<:Symbol}, Symbol}`: Types of ACW computed
- `n_lags::Union{Int, Nothing}`: Number of lags used for ACF calculation
- `freqlims::Union{Tuple{Real, Real}, Nothing}`: Frequency limits used for spectral analysis
- `acf::Union{AbstractVector{<:Real}, AbstractArray{<:Real}, Nothing}`: Autocorrelation function
- `psd::Union{AbstractVector{<:Real}, AbstractArray{<:Real}, Nothing}`: Power spectral density
- `freqs::Union{AbstractVector{<:Real}, AbstractArray{<:Real}, Nothing}`: Frequency vector for PSD
- `lags::Union{AbstractVector{<:Real}, AbstractArray{<:Real}, Nothing}`: Lag vector for ACF
- `x_dim::Union{Int, Nothing}`: Dimension index corresponding to x-axis (lags/freqs)

# Notes
- Supported ACW types: :acw0, :acw50, :acweuler, :auc, :tau, :knee
- Results order matches input acwtypes order
- If only one ACW type is requested, `acw_results` is a scalar; otherwise it's a vector
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

possible_acwtypes = [:acw0, :acw50, :acweuler, :auc, :tau, :knee]

"""
    acw(data, fs; acwtypes=possible_acwtypes, n_lags=nothing, freqlims=nothing, time=nothing, 
        dims=ndims(data), return_acf=true, return_psd=true, average_over_trials=false,
        trial_dims::Int=setdiff([1, 2], dims)[1], skip_zero_lag::Bool=false, max_peaks::Int=1, oscillation_peak::Bool=true,
        allow_variable_exponent::Bool=false)

Compute various timescale measures for time series data. For detailed documentaion, see https://duodenum96.github.io/IntrinsicTimescales.jl/stable/acw/. 

# Arguments
- `data::AbstractArray{<:Real}`: Input time series data
- `fs::Real`: Sampling frequency
- `acwtypes::Union{Vector{Symbol}, Symbol}=[:acw0, :acw50, :acweuler, :auc, :tau, :knee]`: Types of ACW to compute.
Supported ACW types:
  * :acw0 - Time to first zero crossing
  * :acw50 - Time to 50% decay
  * :acweuler - Time to 1/e decay
  * :auc - Area under curve of ACF before ACW0
  * :tau - Exponential decay timescale
  * :knee - Knee frequency from spectral analysis
- `n_lags::Union{Int, Nothing}=nothing`: Number of lags for ACF calculation. If not specified, uses 1.1 * ACW0.
- `freqlims::Union{Tuple{Real, Real}, Nothing}=nothing`: Frequency limits for spectral analysis. If not specified, uses full frequency range.
- `time::Union{Vector{Real}, Nothing}=nothing`: Time vector. This is required for Lomb-Scargle method in the case of missing data.
- `dims::Int=ndims(data)`: Dimension along which to compute ACW (Dimension of time)
- `return_acf::Bool=true`: Whether to return the ACF
- `return_psd::Bool=true`: Whether to return the PSD
- `average_over_trials::Bool=false`: Whether to average the ACF or PSD over trials
- `trial_dims::Int=setdiff([1, 2], dims)[1]`: Dimension along which to average the ACF or PSD over trials (Dimension of trials)
- `skip_zero_lag::Bool=false`: Whether to skip the zero lag for fitting an exponential decay function. Used only for :tau.
- `max_peaks::Int=1`: Maximum number of oscillatory peaks to fit in spectral analysis
- `oscillation_peak::Bool=true`: Whether to fit an oscillation peak in the spectral analysis
- `allow_variable_exponent::Bool=false`: Whether to allow variable exponent in spectral fitting

# Returns
- `ACWResults`: Structure containing computed ACW measures and intermediate results

Fields of the ACWResults structure:
- `fs::Real`: Sampling frequency
- `acw_results`: Computed ACW values (type depends on number of ACW types requested)
- `acwtypes::Union{Vector{<:Symbol}, Symbol}`: Types of ACW computed
- `n_lags::Union{Int, Nothing}`: Number of lags used for ACF calculation
- `freqlims::Union{Tuple{Real, Real}, Nothing}`: Frequency limits used for spectral analysis
- `acf::Union{AbstractVector{<:Real}, AbstractArray{<:Real}, Nothing}`: Autocorrelation function
- `psd::Union{AbstractVector{<:Real}, AbstractArray{<:Real}, Nothing}`: Power spectral density
- `freqs::Union{AbstractVector{<:Real}, AbstractArray{<:Real}, Nothing}`: Frequency vector for PSD
- `lags::Union{AbstractVector{<:Real}, AbstractArray{<:Real}, Nothing}`: Lag vector for ACF
- `x_dim::Union{Int, Nothing}`: Dimension index corresponding to x-axis (lags/freqs)
"""
function acw(data, fs; acwtypes=possible_acwtypes, n_lags=nothing, freqlims=nothing, time=nothing, 
             dims=ndims(data), return_acf=true, return_psd=true, average_over_trials=false,
             trial_dims::Int=setdiff([1, 2], dims)[1], skip_zero_lag::Bool=false, max_peaks::Int=1, oscillation_peak::Bool=true,
             allow_variable_exponent::Bool=false)

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
    acf_acwtypes = [:acw0, :acw50, :acweuler, :auc, :tau]
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
        if any(in.(:auc, [acwtypes]))
            auc_idx = findfirst(acwtypes .== :auc)
            auc_result = acw_romberg(dt, selectdim(acf, dims, 1:floor(Int, acw0_sample[1])); dims=dims)
            if (auc_result isa Vector) && (length(auc_result) == 1)
                result[auc_idx] = auc_result[1]
            else
                result[auc_idx] = auc_result
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
            if !skip_zero_lag
                tau_result = fit_expdecay(collect(lags), acf; dims=dims)
            else
                tau_result = fit_expdecay_3_parameters(collect(lags), acf; dims=dims)
            end
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
            time = dt:dt:(size(data, dims)*dt)
            psd, freqs = comp_psd_lombscargle(time, data, nanmask, dt; dims=dims)
        end

        if average_over_trials
            psd = mean(psd, dims=trial_dims)
        end

        if isnothing(freqlims)
            freqlims = (freqs[1], freqs[end])
        end
        knee_result = tau_from_knee(fooof_fit(psd, freqs; dims=dims, min_freq=freqlims[1], 
                                             max_freq=freqlims[2], oscillation_peak=oscillation_peak, 
                                             max_peaks=max_peaks, return_only_knee=true,
                                             allow_variable_exponent=allow_variable_exponent))
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
        return ACWResults(fs, map(identity, result), acwtypes, n_lags, freqlims, acf, psd, freqs, lags, x_dim)
    end
end


end
