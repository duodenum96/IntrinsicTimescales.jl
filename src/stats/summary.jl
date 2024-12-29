# src/stats/summary.jl
"""
Compute summary statistics for time series data
"""
module SummaryStats
using FFTW, Statistics
import DSP as dsp
import StatsBase as sb

export comp_ac_fft, comp_psd, comp_cc, comp_ac_time

"""
Compute autocorrelation using FFT
"""
function comp_ac_fft(data::AbstractMatrix; normalize::Bool=true, n_lags::Int=3000)
    n = size(data, 2)
    xp = data .- mean(data; dims=2)

    # Zero padding
    xp = hcat(xp, zeros(size(xp)))
    xp = hcat(zeros(size(xp)), xp)

    # FFT computation
    f = fft(xp, 2)
    p = abs2.(f)
    p_i = ifft(p, 2)

    # Extract real part and normalize
    ac_all = real.(p_i)[:, 1:(n-1)] ./ range(n - 1, 1; step=-1)'
    ac = mean(ac_all; dims=1)[:][1:n_lags]

    return normalize ? ac ./ maximum(ac) : ac
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
                  window::Function=dsp.hamming,
                  n=div(length(x), 8),
                  noverlap=div(n, 2))
    if method == "periodogram"
        psd = dsp.periodogram(x; fs=fs, window=window)
    elseif method == "welch"
        @warn "Using Welch method. Don't trust the defaults!"
        psd = dsp.welch_pgram(x, fs; window=window, n=n, noverlap=noverlap)
    else
        error("Invalid method: $method")
    end
    return psd.power, psd.freq
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
    
    lags = 0:max_lag
    n_trials = size(data, 1)
    cc = zeros(n_trials, max_lag + 1)
    for trial in 1:n_trials
        cc[trial, :] = sb.autocor(data[trial, :], lags)
    end
    cc_mean = mean(cc; dims=1)[:]
    return cc_mean
end

end # module