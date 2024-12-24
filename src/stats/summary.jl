# src/stats/summary.jl
"""
Compute summary statistics for time series data
"""
module SummaryStats
using FFTW, Statistics, DSP

export comp_ac_fft, comp_psd, comp_cc

"""
Compute autocorrelation using FFT
"""
function comp_ac_fft(data::AbstractMatrix)
    n = size(data, 2)
    xp = data .- mean(data, dims=2)
    
    # Zero padding
    xp = hcat(xp, zeros(size(xp)))
    
    # FFT computation
    f = fft(xp, 2)
    p = abs2.(f)
    pi = ifft(p, 2)
    
    # Extract real part and normalize
    ac_all = real.(pi)[:, 1:n-1] ./ range(n-1, 1, step=-1)'
    ac = mean(ac_all, dims=1)[:]
    
    return ac
end

"""
Compute power spectral density
"""
function comp_psd(x::AbstractMatrix, T::Float64, deltaT::Float64)
    fs = T/deltaT
    n_points = size(x, 2)
    
    # Apply Hamming window and remove mean
    window = hamming(n_points)
    x_windowed = (x .- mean(x, dims=2)) .* window'
    
    # Compute PSD
    psd = mean(abs2.(rfft(x_windowed, 2)), dims=1)[:]
    
    # Remove DC and Nyquist components
    return psd[2:end-1]
end

"""
Compute cross-correlation in time domain
"""
function comp_cc(data1::AbstractMatrix, data2::AbstractMatrix, max_lag::Int, bin_size::Float64, num_bin::Int)
    num_trials = size(data1, 1)
    cc_sum = zeros(max_lag + 1)
    
    for trial in 1:num_trials
        for lag in 0:max_lag
            idx = 1:(num_bin-lag)
            cc_sum[lag+1] += mean(@view(data1[trial, idx]) .* @view(data2[trial, idx.+lag]))
        end
    end
    
    return cc_sum ./ num_trials
end

end # module