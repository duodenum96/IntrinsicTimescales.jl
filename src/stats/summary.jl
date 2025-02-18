"""
    SummaryStats

Module for computing various summary statistics from time series data.
Includes functions for:
- Autocorrelation (FFT and time-domain methods)
- Power spectral density (periodogram and Welch methods)
- Cross-correlation
- Special handling for missing data (NaN values)
"""
module SummaryStats
# using FFTW, Statistics
using FFTW
using Statistics
using FastTransformsForwardDiff
import DSP as dsp
import StatsBase as sb
using Missings
using LinearAlgebra
import LombScargle as ls
include("bat_autocor.jl")

export comp_ac_fft, comp_psd, comp_cc, comp_ac_time, comp_ac_time_missing,
       comp_ac_time_adfriendly, comp_psd_adfriendly, comp_psd_lombscargle,
       prepare_lombscargle, _comp_psd_lombscargle

"""
    comp_ac_fft(data::Vector{T}; n_lags::Real=length(data)) where {T <: Real}

Compute autocorrelation using FFT method.

# Arguments
- `data`: Input time series vector
- `n_lags`: Number of lags to compute (defaults to length of data)

# Returns
- Vector of autocorrelation values from lag 0 to n_lags-1

# Notes
- Uses FFT for efficient computation
- Pads data to next power of 2 for FFT efficiency
- Normalizes by variance (first lag)
"""
function comp_ac_fft(data::Vector{T}; n_lags::Real=length(data)) where {T <: Real}
    # Center the data
    x = data .- mean(data)
    n = length(x)

    # Pad to next power of 2 for FFT efficiency
    n_pad = nextpow(2, 2n - 1)  # For autocorrelation, need 2n-1 points
    x_pad = vcat(x, zeros(T, n_pad - n))

    # Compute autocorrelation via FFT
    Frf = fft(x_pad)
    acov = real(ifft(abs2.(Frf)))[1:n] ./ n

    # Normalize by variance (first lag)
    ac = acov ./ acov[1]

    return ac[1:n_lags]
end

"""
    comp_ac_fft(data::AbstractArray{T}; dims::Int=ndims(data), n_lags::Integer=size(data, dims)) where {T <: Real}

Compute autocorrelation using FFT along specified dimension.

# Arguments
- `data`: Array of time series data
- `dims`: Dimension along which to compute autocorrelation (defaults to last dimension)
- `n_lags`: Number of lags to compute (defaults to size of data along specified dimension)

# Returns
Array with autocorrelation values, the specified dimension becomes the dimension of lags while the other dimensions denote ACF values
"""
function comp_ac_fft(data::AbstractArray{T}; dims::Real=ndims(data),
                     n_lags::Real=size(data, dims)) where {T <: Real}
    f = x -> comp_ac_fft(vec(x), n_lags=n_lags)
    return mapslices(f, data, dims=dims)
end

"""
    comp_psd(x::AbstractArray{T}, fs::Real; kwargs...) where {T <: Real}

Compute power spectral density using periodogram or welch method.

# Arguments
- `x`: Time series data (time × channels)
- `fs`: Sampling frequency
- `dims=ndims(x)`: Dimension along which to compute PSD
- `method="periodogram"`: Method to use ("periodogram" or "welch")
- `window=dsp.hamming`: Window function
- `n=div(size(x,dims),8)`: Window size for Welch method
- `noverlap=div(n,2)`: Overlap for Welch method

# Returns
- `power`: Power spectral density values
- `freqs`: Corresponding frequencies

# Notes
- For Welch method, carefully consider window size and overlap
- Uses DSP.jl for underlying computations
"""
function comp_psd(x::AbstractArray{T}, fs::Real;
                  dims::Int=ndims(x),
                  method::String="periodogram",
                  window=dsp.hamming,
                  n=div(size(x, dims), 8),
                  noverlap=div(n, 2)) where {T <: Real}
    # Create a wrapper function that only returns power
    f = x -> begin
        power, _ = comp_psd(vec(x), fs, method=method, window=window, n=n,
                            noverlap=noverlap)
        return power
    end

    # Apply the function along the specified dimension
    power = mapslices(f, x, dims=dims)

    # Get a single time series for frequency calculation
    # Create indices to get first element along all dimensions except dims
    # TODO: This is redundant, we are already getting indices above. 
    idx = [i == dims ? (1:size(x, dims)) : 1 for i in 1:ndims(x)]
    first_slice = x[idx...]

    # Compute frequencies once since they're the same for all slices
    _, freqs = comp_psd(vec(first_slice), fs,
                        method=method, window=window, n=n, noverlap=noverlap)

    return power, freqs
end

function comp_psd(x::Vector{T}, fs::Real;
                  method::String="periodogram",
                  window=dsp.hamming,
                  n=div(length(x), 8),
                  noverlap=div(n, 2)) where {T <: Real}
    if method == "periodogram"
        psd = dsp.periodogram(x; fs=fs, window=window)
        power = psd.power[2:end]
        freqs = psd.freq[2:end]
    elseif method == "welch"
        @warn "Using Welch method. Don't trust the defaults!"
        psd = dsp.welch_pgram(x, n, noverlap; fs=fs, window=window)
        power = psd.power
        freqs = psd.freq
    else
        error("Invalid method: $method")
    end
    return power, freqs
end

"""
    comp_psd_adfriendly(x::AbstractArray{<:Real}, fs::Real; dims::Int=ndims(x))

Compute power spectral density using an automatic differentiation (AD) friendly implementation.

# Arguments
- `x`: Time series data
- `fs`: Sampling frequency
- `dims=ndims(x)`: Dimension along which to compute PSD

# Returns
- `power`: Power spectral density values
- `freqs`: Corresponding frequencies
"""
function comp_psd_adfriendly(x::AbstractArray{<:Real}, fs::Real; dims::Int=ndims(x))
    f = x -> comp_psd_adfriendly(vec(x), fs)[1]
    power = mapslices(f, x, dims=dims)

    # Get a single time series for frequency calculation
    idx = [i == dims ? (1:size(x, dims)) : 1 for i in 1:ndims(x)]
    first_slice = x[idx...]

    # Compute frequencies once
    _, freqs = comp_psd_adfriendly(vec(first_slice), fs)

    return power, freqs
end

function comp_psd_adfriendly(x::Vector{<:Real}, fs::Real; demean::Bool=true)
    n = length(x)
    n2 = 2 * _ac_next_pow_two(n)
    
    # Prepare the data
    if demean
        x_demeaned = x .- mean(x)
    else
        x_demeaned = x
    end

    # Compute window (Hamming)
    window = 0.54 .- 0.46 .* cos.(2π .* (0:n-1) ./ (n - 1))
    
    # Zero pad the windowed data
    x_padded = zeros(eltype(x), n2)
    x_padded[1:n] = x_demeaned .* window
    
    # Scale factor for power normalization
    # Using n instead of n2 because that's our effective data length
    scale = 1.0 / (fs * sum(window .^ 2))
    
    # Compute FFT and get power
    x_fft = fft(x_padded)
    psd = abs2.(x_fft[1:div(n2,2)]) .* scale
    
    # Compute frequency vector (only positive frequencies)
    freqs = fftfreq(n2, fs)[1:div(n2,2)]
    
    # Return only positive frequencies, excluding DC (zero frequency)
    return psd[2:end], freqs[2:end]
end

"""
    _comp_psd_lombscargle(times, data, frequency_grid)

Internal function to compute Lomb-Scargle periodogram for a single time series.

# Arguments
- `times`: Time points vector (without NaN)
- `data`: Time series data (without NaN)
- `frequency_grid`: Pre-computed frequency grid

# Returns
- `power`: Lomb-Scargle periodogram values
- `frequency_grid`: Input frequency grid

# Notes
- Uses LombScargle.jl for core computation
- Assumes data has been pre-processed and doesn't contain NaN values
- Normalizes power spectrum by variance
"""
function _comp_psd_lombscargle(times::AbstractVector{<:Number},
                               signal::AbstractVector{<:Number},
                               frequency_grid::AbstractVector{<:Number})
    plan = ls.plan(times, signal, frequencies=frequency_grid)
    psd = ls.lombscargle(plan)
    return psd.power
end

"""
    prepare_lombscargle(times, data, nanmask)

Prepare data for Lomb-Scargle periodogram computation by handling missing values.

# Arguments
- `times`: Time points vector
- `data`: Time series data (may contain NaN)
- `nanmask`: Boolean mask indicating NaN positions

# Returns
- `valid_times`: Time points with NaN values removed
- `valid_data`: Data points with NaN values removed
- `frequency_grid`: Suggested frequency grid for analysis
"""
function prepare_lombscargle(times::AbstractVector{T}, data::AbstractMatrix{S},
                             nanmask::AbstractMatrix{Bool}, dt::Real) where {T<:Number, S<:Number}
    n_trials = size(data, 1)
    times_masked = Vector{Vector{T}}(undef, n_trials)
    signal_masked = Vector{Vector{S}}(undef, n_trials)
    for i in 1:n_trials
        times_masked[i] = times[.!nanmask[i, :]]
        signal_masked[i] = data[i, .!nanmask[i, :]]
    end
    # Create a common frequency grid based on the shortest time series
    # (most conservative approach to avoid aliasing)
    shortest_times = times_masked[argmin(length.(times_masked))]
    frequency_grid = ls.autofrequency(shortest_times, maximum_frequency=1 / (2dt))

    return times_masked, signal_masked, frequency_grid
end

"""
    comp_psd_lombscargle(times, data, nanmask, dt; dims=ndims(data))

Compute Lomb-Scargle periodogram for data with missing values.

# Arguments
- `times`: Time points vector
- `data`: Time series data (may contain NaN)
- `nanmask`: Boolean mask indicating NaN positions
- `dt`: Time step
- `dims=ndims(data)`: Dimension along which to compute

# Returns
- `power`: Lomb-Scargle periodogram values
- `frequency_grid`: Corresponding frequencies

# Notes
- Handles irregular sampling due to missing data
- Uses frequency grid based on shortest valid time series
- Automatically determines appropriate frequency range
"""
function comp_psd_lombscargle(times::AbstractVector{<:Number}, data::AbstractVector{<:Number},
                              nanmask::AbstractVector{Bool}, dt::Real)
    times_masked = times[.!nanmask]
    signal_masked = data[.!nanmask]
    frequency_grid = ls.autofrequency(times_masked, maximum_frequency=1 / (2dt))
    psd = _comp_psd_lombscargle(times_masked, signal_masked, frequency_grid)
    return psd, frequency_grid
end

function comp_psd_lombscargle(times::AbstractVector{T}, data::AbstractArray{S},
                              nanmask::AbstractArray{Bool}, dt::Real; dims::Int=ndims(data)) where {T<:Number, S<:Number}
    # Get output size for pre-allocation
    frequency_grid = ls.autofrequency(times, maximum_frequency=1 / (2dt))
    nfreq = length(frequency_grid)
    output_size = collect(size(data))
    output_size[dims] = nfreq
    power = Array{promote_type(T,S)}(undef, output_size...)

    # Create indices for iterating over all dimensions except dims
    other_dims = setdiff(1:ndims(data), dims)
    ranges = [1:size(data, d) for d in other_dims]

    # Iterate over all other dimensions
    for idx in Iterators.product(ranges...)
        # Create index for the full array
        full_idx = [i == dims ? (1:size(data, dims)) : idx[findfirst(==(i), other_dims)]
                    for i in 1:ndims(data)]

        # Extract the time series and its mask
        series = data[full_idx...]
        mask = nanmask[full_idx...]

        # Compute PSD for this slice
        psd = _comp_psd_lombscargle(times[.!mask], vec(series)[.!mask], frequency_grid)

        # Store the result
        full_idx[dims] = 1:nfreq
        power[full_idx...] = psd
    end

    return power, frequency_grid
end

"""
    comp_cc(data1::AbstractArray{T}, data2::AbstractArray{T}, max_lag::Integer;
           dims::Int=ndims(data1)) where {T <: Real}

Compute cross-correlation between two arrays along specified dimension.

# Arguments
- `data1`: First array of time series data
- `data2`: Second array of time series data
- `max_lag`: Maximum lag to compute
- `dims`: Dimension along which to compute cross-correlation (defaults to last dimension)

# Returns
Array with cross-correlation values, reduced along specified dimension
"""
function comp_cc(data1::Vector{T}, data2::Vector{T}, max_lag::Integer) where {T <: Real}
    num_bin = length(data1)
    cc = zeros(T, max_lag + 1)

    for lag in 0:max_lag
        idx = 1:(num_bin-lag)
        cc[lag+1] = mean(@view(data1[idx]) .* @view(data2[idx.+lag]))
    end

    return cc
end

function comp_cc(data1::AbstractArray{T}, data2::AbstractArray{T}, max_lag::Integer;
                 dims::Int=ndims(data1)) where {T <: Real}
    f = x -> comp_cc(vec(x), vec(selectdim(data2, dims, 1:size(x, dims))), max_lag)
    dropdims(mapslices(f, data1, dims=dims), dims=dims)
end

"""
    comp_ac_time(data::AbstractArray{T}, max_lag::Integer; dims::Int=ndims(data)) where {T <: Real}

Compute autocorrelation in time domain along specified dimension.

# Arguments
- `data`: Array of time series data
- `max_lag`: Maximum lag to compute
- `dims`: Dimension along which to compute autocorrelation (defaults to last dimension)

# Returns
Array with autocorrelation values, the specified dimension becomes the dimension of lags while the other dimensions denote ACF values
"""
function comp_ac_time(data::Vector{T}; n_lags::Integer=length(data)) where {T <: Real}
    lags = 0:(n_lags-1)
    sb.autocor(data, lags)
end

function comp_ac_time(data::AbstractArray{T}; dims::Int=ndims(data),
                      n_lags::Integer=size(data, dims)) where {T <: Real}
    f = x -> comp_ac_time(vec(x), n_lags=n_lags)
    return mapslices(f, data, dims=dims)
end

"""
    comp_ac_time_missing(data::AbstractArray{T}; kwargs...) where {T <: Real}

Compute autocorrelation for data with missing values.

# Arguments
- `data`: Time series data (may contain NaN)
- `dims=ndims(data)`: Dimension along which to compute
- `n_lags=size(data,dims)`: Number of lags to compute

# Returns
- Array of autocorrelation values

# Notes
- Handles missing data using "conservative" approach
- Sets NaN values to zero after mean adjustment
- Returns NaN for lags with insufficient valid pairs
- Based on statsmodels.tsa.stattools implementation
"""
function comp_ac_time_missing(data::AbstractVector{T}; n_lags::Integer=length(data)) where {T <: Real}
    lags = collect(0:(n_lags-1))
    ac = zeros(T, n_lags)
    n = length(data)

    # Handle missing values
    notmask_bool = .!isnan.(data)
    
    # Must copy for thread safety
    x = copy(data)
    x[.!notmask_bool] .= 0  # Set NaNs to 0 like in acovf

    # Center the data using only non-missing values
    xm = sum(x) / sum(notmask_bool)
    x .-= xm

    # Compute denominator (sum of squares normalized by n)
    ss = sum(abs2, x) / n

    for lag in lags
        # Compute autocovariance using masked values
        # The zeros we inserted for NaNs will not contribute to the sum
        ac[lag+1] = sum(view(x, 1:(n-lag)) .* view(x, (lag+1):n)) / n
    end

    # Normalize by variance to get correlation
    ac ./= ss

    return ac
end

function comp_ac_time_missing(data::AbstractArray{T}; dims::Int=ndims(data),
                              n_lags::Integer=size(data, dims)) where {T <: Real}
    f = x -> comp_ac_time_missing(vec(x), n_lags=n_lags)
    return mapslices(f, data, dims=dims)
end

# The two functions below are Julia translations of the functions from statsmodels.tsa.stattools

"""
    acf_statsmodels(x::Vector{T}; kwargs...) where {T <: Real}

Julia implementation of statsmodels.tsa.stattools.acf function. Only for testing.

# Arguments
- `x`: Time series data vector
- `adjusted=false`: Use n-k denominators if true
- `nlags=nothing`: Number of lags (default: min(10*log10(n), n-1))
- `qstat=false`: Return Ljung-Box Q-statistics
- `isfft=false`: Use FFT method
- `alpha=nothing`: Confidence level for intervals
- `bartlett_confint=false`: Use Bartlett's formula
- `missing_handling="conservative"`: NaN handling method

# Returns
- Vector of autocorrelation values

# Notes
- Supports multiple missing data handling methods:
  - "none": No checks
  - "raise": Error on NaN
  - "conservative": NaN-aware computations
  - "drop": Remove NaN values
"""
function acf_statsmodels(x::Vector{T};
                         adjusted::Bool=false,
                         nlags::Union{Int, Nothing}=nothing,
                         qstat::Bool=false,
                         isfft::Bool=false,
                         alpha::Union{Float64, Nothing}=nothing,
                         bartlett_confint::Bool=false,
                         missing_handling::String="conservative") where {T <: Real}

    # Input validation
    missing_handling = lowercase(missing_handling)
    @assert missing_handling in ["none", "raise", "conservative", "drop"]

    nobs = length(x)
    if isnothing(nlags)
        nlags = min(Int(floor(10 * log10(nobs))), nobs - 1)
    end

    # Compute autocovariance
    avf = acovf(x; adjusted=adjusted, demean=true, isfft=isfft,
                missing_handling=missing_handling, nlag=nlags)
    acf_vals = avf[1:nlags+1] ./ avf[1]

    return acf_vals
end

"""
Estimate autocovariances.
Translated to Julia from statsmodels.tsa.stattools

Parameters
----------
x : array_like
    Time series data. Must be 1d.
adjusted : bool, default False
    If True, then denominators is n-k, otherwise n.
demean : bool, default True
    If True, then subtract the mean x from each element of x.
fft : bool, default True
    If True, use FFT convolution.  This method should be preferred
    for long time series.
missing : str, default "none"
    A string in ["none", "raise", "conservative", "drop"] specifying how
    the NaNs are to be treated. "none" performs no checks. "raise" raises
    an exception if NaN values are found. "drop" removes the missing
    observations and then estimates the autocovariances treating the
    non-missing as contiguous. "conservative" computes the autocovariance
    using nan-ops so that nans are removed when computing the mean
    and cross-products that are used to estimate the autocovariance.
    When using "conservative", n is set to the number of non-missing
    observations.
nlag : {int, None}, default None
    Limit the number of autocovariances returned.  Size of returned
    array is nlag + 1.  Setting nlag when fft is False uses a simple,
    direct estimator of the autocovariances that only computes the first
    nlag + 1 values. This can be much faster when the time series is long
    and only a small number of autocovariances are needed.

Returns
-------
ndarray
    The estimated autocovariances.

References
----------
.. [1] Parzen, E., 1963. On spectral analysis with missing observations
       and amplitude modulation. Sankhya: The Indian Journal of
       Statistics, Series A, pp.383-392.
"""
function acovf(x::Vector{T};
               adjusted::Bool=false,
               demean::Bool=true,
               isfft::Bool=false,
               missing_handling::String="none",
               nlag::Union{Int, Nothing}=nothing) where {T <: Real}

    # Input validation
    missing_handling = lowercase(missing_handling)
    @assert missing_handling in ["none", "raise", "conservative", "drop"]

    # Handle missing values
    deal_with_masked = missing_handling != "none" && any(isnan.(x))
    if deal_with_masked
        if missing_handling == "raise"
            throw(ArgumentError("NaNs were encountered in the data"))
        end
        notmask_bool = .!isnan.(x)
        if missing_handling == "conservative"
            # Must copy for thread safety
            x = copy(x)
            x[.!notmask_bool] .= 0
        else # "drop"
            x = x[notmask_bool]
        end
        notmask_int = Int.(notmask_bool)
    end

    # Demean the data
    if demean && deal_with_masked
        xo = map(v -> isnan(v) ? 0 : v - sum(x[notmask_bool]) / sum(notmask_int), x) # if nan, replace with 0, else, demean
    elseif demean
        xo = x .- mean(x)
    else
        xo = x
    end

    n = length(x)
    lag_len = isnothing(nlag) ? n - 1 : nlag
    if !isnothing(nlag) && nlag > n - 1
        throw(ArgumentError("nlag must be smaller than nobs - 1"))
    end

    if !isfft && !isnothing(nlag)
        acov = [i == 0 ?
                dot(xo, xo) :
                dot(view(xo, (i+1):length(xo)), view(xo, 1:(length(xo)-i)))
                for i in 0:lag_len]
        if !deal_with_masked || missing_handling == "drop"
            if adjusted
                acov2 = acov ./ (n .- (0:lag_len))
            else
                acov2 = acov ./ n
            end
        else
            if adjusted
                divisor = zeros(Int, lag_len + 1)
                divisor[1] = sum(notmask_int)
                for i in 1:lag_len
                    divisor[i+1] = dot(view(notmask_int, (i+1):length(notmask_int)),
                                       view(notmask_int, 1:(length(notmask_int)-i)))
                end
                divisor[divisor.==0] .= 1
                acov2 = acov ./ divisor
            else # biased, missing data but not "drop"
                acov2 = acov ./ sum(notmask_int)
            end
        end
        return acov2
    end

    if adjusted && deal_with_masked && missing_handling == "conservative"
        d = dsp.xcorr(notmask_int, notmask_int)
        d[d.==0] .= 1
    elseif adjusted
        xi = 1:n
        d = vcat(xi, xi[end-1:-1:1])
    elseif deal_with_masked
        # biased and NaNs given and ("drop" or "conservative")
        d = sum(notmask_int) * ones(2n - 1)
    else # biased and no NaNs or missing=="none"
        d = n * ones(2n - 1)
    end

    if isfft
        nobs = length(xo)
        n = nextpow(2, 2nobs + 1)
        Frf = fft(xo, n)
        acov = real(ifft(Frf .* conj(Frf)))[1:nobs] ./ d[nobs:end]
    else
        acov = DSP.xcorr(xo, xo)[n:end] ./ d[n:end]
    end

    if !isnothing(nlag)
        # Copy to allow gc of full array rather than view
        return copy(acov[1:lag_len+1])
    end
    return acov
end

end # module