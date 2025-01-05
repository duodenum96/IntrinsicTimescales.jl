# src/utils/utils.jl
module Utils
using LsqFit
export expdecayfit, find_oscillation_peak

"""
Fit an exponential decay to the data
p[1] * exp(-p[2] * data)
data is a 1D vector (ACF)
lags is a 1D vector (x axis)
"""
function expdecayfit(data, lags)
    # Return best fit parameters
    return fit.param
end

"""
Find the dominant oscillatory peak in the PSD using prominence
"""
function find_oscillation_peak(psd::Vector{Float64}, freqs::Vector{Float64};
                             min_freq::Float64=5.0 / 1000.0,
                             max_freq::Float64=50.0 / 1000.0,
                             min_prominence_ratio::Float64=0.1)  # minimum prominence as fraction of max PSD
    # Consider only frequencies in the specified range
    freq_mask = (freqs .>= min_freq) .& (freqs .<= max_freq)
    search_psd = psd[freq_mask]
    search_freqs = freqs[freq_mask]
    
    # Find peaks
    peak_indices = Int[]
    for i in 2:length(search_psd)-1
        if search_psd[i] > search_psd[i-1] && search_psd[i] > search_psd[i+1]
            push!(peak_indices, i)
        end
    end
    
    # If no peaks found, return NaN
    if isempty(peak_indices)
        return NaN
    end
    
    # Calculate prominence for each peak
    prominences = Float64[]
    for idx in peak_indices
        # Find higher of the two minima on either side of peak
        left_min = minimum(search_psd[1:idx])
        right_min = minimum(search_psd[idx:end])
        base = max(left_min, right_min)
        
        # Prominence is height above this base
        prominence = search_psd[idx] - base
        push!(prominences, prominence)
    end
    
    # Filter peaks by minimum prominence
    min_prominence = maximum(search_psd) * min_prominence_ratio
    valid_peaks = peak_indices[prominences .>= min_prominence]
    valid_prominences = prominences[prominences .>= min_prominence]
    
    # If no peaks meet prominence criterion, return NaN
    if isempty(valid_peaks)
        return NaN
    end
    
    # Return frequency of the most prominent peak
    best_peak_idx = valid_peaks[argmax(valid_prominences)]
    return search_freqs[best_peak_idx]
end

end # module