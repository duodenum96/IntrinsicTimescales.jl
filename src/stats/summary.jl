# src/stats/summary.jl
"""
Compute summary statistics for time series data
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
using Infiltrator
include("bat_autocor.jl")

export comp_ac_fft, comp_psd, comp_cc, comp_ac_time, comp_ac_time_missing,
       comp_ac_time_adfriendly, comp_psd_adfriendly, comp_psd_lombscargle,
       prepare_lombscargle, _comp_psd_lombscargle

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
Compute power spectral density
(Just a wrapper for DSP.jl)
Arguments:
- x: time series data (time X channels)
- fs: sampling frequency
Optional arguments:
- method: method to use for PSD computation ("periodogram" or "welch")
- window: window to use for PSD computation (default is hamming)
- n: window size for Welch method (default is 1/8 of the length of x)
- noverlap: overlap for Welch method (default is 1/2 of the window size)
Returns:
- psd: power spectral density
- freq: frequency

See the documentation of DSP.jl for more details.
NOTE: If you are using Welch method, don't trust the defaults! Make sure you have reasonable 
overlap and window size.
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
    x2 = zeros(eltype(x), n2)
    idxs2 = firstindex(x2):(firstindex(x2)+n-1)

    if demean
        x2_demeaned = x .- mean(x)
    else
        x2_demeaned = x
    end

    # Compute window (Hamming)
    window = 0.54 .- 0.46 .* cos.(2π .* (0:n-1) ./ (n - 1))

    # Scale factor for power normalization
    scale = 1.0 / (fs * sum(window .^ 2))

    x2_fft = fft(x2_demeaned .* window)
    psd = real.(view(x2_fft .* conj.(x2_fft), idxs2)) .* scale

    freqs = fftfreq(n2, fs)[1:n]
    freqs2 = freqs[freqs.≥0]

    return psd[2:end], freqs2[2:end]
end

"""
Lomb-Scargle periodogram for time series data with missing values
"""
function _comp_psd_lombscargle(times::AbstractVector{<:Real},
                               signal::AbstractVector{<:Real},
                               frequency_grid::AbstractVector{<:Real})
    plan = ls.plan(times, signal, frequencies=frequency_grid)
    psd = ls.lombscargle(plan)
    return psd.power
end

"""
    prepare_lombscargle(times::AbstractVector{<:Real}, data::AbstractMatrix{<:Real}, nanmask::AbstractMatrix{Bool})

Prepare time series data with missing values (NaNs) for Lomb-Scargle periodogram analysis by masking out NaN values.
    Challange: Each trial may have a different number of NaNs.
    The way we deal with is going from matrix to a vector of vectors.
    Each vector is a trial with NaNs removed.

# Arguments
- `times`: Vector of time points
- `data`: Matrix of time series data, with trials as rows and time points as columns
- `nanmask`: Boolean matrix indicating location of NaN values (true where NaN)

# Returns
- `times_masked`: Vector of vectors of time points with NaN values removed (each vector is a trial)
- `signal_masked`: Vector of vectors of data values with NaN values removed (each vector is a trial)
"""
function prepare_lombscargle(times::Vector{Float64}, data::AbstractMatrix{Float64},
                             nanmask::AbstractMatrix{Bool}, dt::Float64)
    n_trials = size(data, 1)
    times_masked = Vector{Vector{Float64}}(undef, n_trials)
    signal_masked = Vector{Vector{Float64}}(undef, n_trials)
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
    comp_psd_lombscargle(times::AbstractVector{<:Real}, data::AbstractArray{<:Real},
                        nanmask::AbstractArray{Bool}, dt::Real; dims::Int=ndims(data))

Compute Lomb-Scargle periodogram for data with missing values along specified dimension.

# Arguments
- `times`: Vector of time points
- `data`: Array of time series data
- `nanmask`: Boolean array indicating missing values (true where NaN)
- `dt`: Time step
- `dims`: Dimension along which to compute PSD (defaults to last dimension)

# Returns
Tuple of (power, frequencies)
"""
function comp_psd_lombscargle(times::AbstractVector{<:Real}, data::AbstractVector{<:Real},
                              nanmask::AbstractVector{Bool}, dt::Real)
    times_masked = times[.!nanmask]
    signal_masked = data[.!nanmask]
    frequency_grid = ls.autofrequency(times_masked, maximum_frequency=1 / (2dt))
    psd = _comp_psd_lombscargle(times_masked, signal_masked, frequency_grid)
    return psd, frequency_grid
end

function comp_psd_lombscargle(times::AbstractVector{<:Real}, data::AbstractArray{<:Real},
                              nanmask::AbstractArray{Bool}, dt::Real; dims::Int=ndims(data))
    # Get output size for pre-allocation
    frequency_grid = ls.autofrequency(times, maximum_frequency=1 / (2dt))
    nfreq = length(frequency_grid)
    output_size = collect(size(data))
    output_size[dims] = nfreq
    power = Array{Float64}(undef, output_size...)

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
    comp_ac_time_missing(data::AbstractArray{T}, max_lag::Integer;
                        dims::Int=ndims(data)) where {T <: Real}

Compute autocorrelation for data with missing values along specified dimension.

# Arguments
- `data`: Array of time series data (can contain NaN)
- `max_lag`: Maximum lag to compute
- `dims`: Dimension along which to compute autocorrelation (defaults to last dimension)
- `min_pairs`: Minimum number of valid pairs required to compute correlation (default: 3)

# Returns
Array with autocorrelation values, specified dimension becomes the dimension for lags. Returns NaN for 
lags with insufficient valid pairs.
"""
function comp_ac_time_missing(data::Vector{T}; n_lags::Integer=length(data),
                              min_pairs::Integer=3) where {T <: Real}
    lags = collect(0:(n_lags-1))
    ac = zeros(T, n_lags)
    n = length(data)

    # Handle missing values
    notmask_bool = .!isnan.(data)

    # Must copy for thread safety
    x = copy(data)

    # Center the data
    xm = mean(view(x, notmask_bool))
    x .-= xm

    # Pre-allocate vectors for valid pairs
    x_valid = Vector{T}(undef, n)
    y_valid = Vector{T}(undef, n)

    # Compute denominator (sum of squares)
    ss = sum(abs2, view(x, notmask_bool))

    for lag in lags
        # Count and collect valid pairs
        # TODO: Like acovf implementation below, set NaNs to 0 to avoid the if loop. 
        valid_count = 0
        for i in 1:(n-lag)
            if notmask_bool[i] && notmask_bool[i + lag]
                valid_count += 1
                x_valid[valid_count] = x[i]
                y_valid[valid_count] = x[i + lag]
            end
        end

        # Compute autocovariance if enough valid pairs
        if valid_count ≥ min_pairs
            x_slice = view(x_valid, 1:valid_count)
            y_slice = view(y_valid, 1:valid_count)
            # Use same normalization as acovf
            ac[lag+1] = sum(x_slice .* y_slice) / n
        else
            ac[lag+1] = NaN
        end
    end

    # Normalize by variance (sum of squares / n) to get correlation
    ac ./= (ss / n)

    return ac
end

function comp_ac_time_missing(data::AbstractArray{T}; dims::Int=ndims(data),
                              n_lags::Integer=size(data, dims),
                              min_pairs::Integer=3) where {T <: Real}
    f = x -> comp_ac_time_missing(vec(x), n_lags=n_lags, min_pairs=min_pairs)
    return mapslices(f, data, dims=dims)
end

# The two functions below are Julia translations of the functions from statsmodels.tsa.stattools

"""
Calculate the autocorrelation function.

Parameters
----------
x : AbstractVector{T} where T<:Real
   The time series data.
adjusted : Bool, default false
   If true, then denominators for autocovariance are n-k, otherwise n.
nlags : Union{Int,Nothing}, default nothing
    Number of lags to return autocorrelation for. If not provided,
    uses min(10 * log10(nobs), nobs - 1). The returned value
    includes lag 0 (ie., 1) so size of the acf vector is (nlags + 1,).
qstat : Bool, default false
    If true, returns the Ljung-Box q statistic for each autocorrelation
    coefficient. See q_stat for more information.
isfft : Bool, default true
    If true, computes the ACF via FFT.
alpha : Union{Float64,Nothing}, default nothing
    If a number is given, the confidence intervals for the given level are
    returned. For instance if alpha=0.05, 95% confidence intervals are
    returned where the standard deviation is computed according to
    Bartlett's formula.
bartlett_confint : Bool, default true
    If true, use Bartlett's formula for confidence intervals.
missing_handling : String, default "none"
    A string in ["none", "raise", "conservative", "drop"] specifying how
    the NaNs are to be treated.

Returns
-------
Tuple containing some combination of:
- acf : Vector{Float64} - The autocorrelation function
- confint : Matrix{Float64} - Confidence intervals (if alpha provided)
- qstat : Vector{Float64} - Q-statistics (if qstat=true)
- pvalue : Vector{Float64} - P-values (if qstat=true)
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