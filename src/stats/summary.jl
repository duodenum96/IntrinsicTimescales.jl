# src/stats/summary.jl
"""
Compute summary statistics for time series data
"""
module SummaryStats
using FFTW, Statistics
import DSP as dsp
import StatsBase as sb
using Missings
using LinearAlgebra
using Infiltrator
include("bat_autocor.jl")

export comp_ac_fft, comp_psd, comp_cc, comp_ac_time, comp_ac_time_missing

"""
Compute autocorrelation using FFT
"""
function comp_ac_fft(data::AbstractMatrix; n_lags::Int=size(data, 2))
    ac = bat_autocorr(data)

    return mean(ac; dims=1)[1:n_lags][:]
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
function comp_psd(x,
                  fs::Float64;
                  method::String="periodogram",
                  window=dsp.hamming,
                  n=div(length(x), 8),
                  noverlap=div(n, 2))
    n_trials = size(x, 1)
    if method == "periodogram"
        if n_trials == 1
            psd = dsp.periodogram(x; fs=fs, window=window)
            power = psd.power
        else
            nfft = dsp.nextfastfft(size(x, 2))
            power = zeros(Int(nfft / 2))
            for i in 1:n_trials
                psd = dsp.periodogram(x[i, :]; fs=fs, window=window)
                power += psd.power[2:end]
            end
            power /= n_trials
        end
    elseif method == "welch"
        @warn "Using Welch method. Don't trust the defaults!"
        psd = dsp.welch_pgram(x, fs; window=window, n=n, noverlap=noverlap)
    else
        error("Invalid method: $method")
    end
    return power, psd.freq
end

"""
Compute cross-correlation in time domain
"""
function comp_cc(data1::AbstractMatrix,
                 data2::AbstractMatrix,
                 max_lag::Int,
                 num_bin::Int)
    num_trials = size(data1, 1)
    cc_sum = zeros(max_lag + 1)

    for trial in 1:num_trials
        for lag in 0:max_lag
            idx = 1:(num_bin-lag)
            cc_sum[lag+1] += mean(@view(data1[trial, idx]) .* @view(data2[trial, idx.+lag]))
        end
    end

    cc_mean = cc_sum ./ num_trials

    return cc_mean
end

function comp_ac_time(data::AbstractMatrix,
                      max_lag::Int)
    
    lags = 0:(max_lag-1)
    n_trials = size(data, 1)
    cc = zeros(n_trials, max_lag)
    for trial in 1:n_trials
        cc[trial, :] = sb.autocor(data[trial, :], lags)
    end
    cc_mean = mean(cc; dims=1)[:]
    return cc_mean
end

function comp_ac_time_missing(data::AbstractMatrix,
                              max_lag::Int)
    n_trials = size(data, 1)
    cc = zeros(n_trials, max_lag)
    for trial in 1:n_trials
        cc[trial, :] = acf_statsmodels(data[trial, :], nlags=max_lag-1)
    end
    cc_mean = mean(cc; dims=1)[:]
    return cc_mean
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
function acf_statsmodels(x::Union{Vector{Float64}, Vector{Union{Missing, Float64}}};
            adjusted::Bool=false,
            nlags::Union{Int,Nothing}=nothing,
            qstat::Bool=false,
            isfft::Bool=false,
            alpha::Union{Float64,Nothing}=nothing,
            bartlett_confint::Bool=false,
            missing_handling::String="conservative")

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
function acovf(x::Union{Vector{Float64}, Vector{Union{Missing, Float64}}};
               adjusted::Bool=false,
               demean::Bool=true,
               isfft::Bool=false,
               missing_handling::String="none",
               nlag::Union{Int,Nothing}=nothing)

    # Input validation
    missing_handling = lowercase(missing_handling)
    @assert missing_handling in ["none", "raise", "conservative", "drop"]

    # Handle missing values
    deal_with_masked = missing_handling != "none" && any(ismissing.(x))
    if deal_with_masked
        if missing_handling == "raise"
            throw(ArgumentError("NaNs were encountered in the data"))
        end
        notmask_bool = .!ismissing.(x)
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
        # whether "drop" or "conservative"
        xo = x .- sum(x)/sum(notmask_int)
        if missing_handling == "conservative"
            xo[.!notmask_bool] .= 0
        end
    elseif demean
        xo = x .- mean(x)
    else
        xo = x
    end

    n = length(x)
    lag_len = isnothing(nlag) ? n-1 : nlag
    if !isnothing(nlag) && nlag > n-1
        throw(ArgumentError("nlag must be smaller than nobs - 1"))
    end

    if !isfft && !isnothing(nlag)
        acov = zeros(Union{Missing, Float64}, lag_len + 1)
        acov[1] = dot(xo, xo)
        for i in 1:lag_len
            acov[i+1] = dot(view(xo, (i+1):length(xo)), view(xo, 1:(length(xo)-i)))
        end
        if !deal_with_masked || missing_handling == "drop"
            if adjusted
                acov ./= (n .- (0:lag_len))
            else
                acov ./= n
            end
        else
            if adjusted
                divisor = zeros(Int, lag_len + 1)
                divisor[1] = sum(notmask_int)
                for i in 1:lag_len
                    divisor[i+1] = dot(view(notmask_int, (i+1):length(notmask_int)),
                                     view(notmask_int, 1:(length(notmask_int)-i)))
                end
                divisor[divisor .== 0] .= 1
                acov ./= divisor
            else # biased, missing data but not "drop"
                acov ./= sum(notmask_int)
            end
        end
        return acov
    end

    if adjusted && deal_with_masked && missing_handling == "conservative"
        d = dsp.xcorr(notmask_int, notmask_int)
        d[d .== 0] .= 1
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