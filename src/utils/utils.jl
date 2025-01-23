# src/utils/utils.jl

module Utils

using Statistics
using NonlinearSolve
export expdecayfit, find_oscillation_peak, find_knee_frequency, fooof_fit,
       lorentzian_initial_guess, lorentzian, expdecay, residual_expdecay!, fit_expdecay,
       acw50, acw50_analytical, acw0, tau_from_acw50

"""
Exponential decay fit
tau = u[1]
acf = exp(-(1/tau) * lags)
acf is a 1D vector (ACF)
lags is a 1D vector (x axis)
"""
function expdecay(tau, lags)
    # Return best fit parameters
    return exp.(-(1 / tau) * lags)
end

"""
Residual function for expdecay
du: residual
u: parameters
p: data
"""
function residual_expdecay!(du, u, p)
    du .= mean(abs2.(expdecay(u[1], p[1]) .- p[2]))
    return nothing
end

function fit_expdecay(lags::Vector{T}, acf::Vector{T}) where {T <: Real}
    u0 = [tau_from_acw50(acw50(lags, acf))]
    prob = NonlinearLeastSquaresProblem(NonlinearFunction(residual_expdecay!,
                                                          resid_prototype=zeros(1)), u0,
                                        p=[lags, acf])
    sol = NonlinearSolve.solve(prob, FastShortcutNLLSPolyalg(), reltol=0.001, verbose=false) # TODO: Find a reasonable tolerance. 
    return sol.u[1]
end

function fit_expdecay(lags::Vector{T}, acf::AbstractArray{T}; dims::Int=ndims(acf)) where {T <: Real}
    f = x -> fit_expdecay(lags, vec(x))
    return dropdims(mapslices(f, acf, dims=dims), dims=dims)
end

acw50_analytical(tau) = -tau * log(0.5)
tau_from_acw50(acw50) = -acw50 / log(0.5)

"""
    acw50(lags::Vector{T}, acf::AbstractArray{T}; dims::Int=ndims(acf)) where {T <: Real}

Compute the ACW50 (autocorrelation width at 50%) along specified dimension.

# Arguments
- `lags`: Vector of lag values
- `acf`: Array of autocorrelation values
- `dims`: Dimension along which to compute ACW50 (defaults to last dimension)

# Returns
Array with ACW50 values, reduced along specified dimension
"""
function acw50(lags::Vector{T}, acf::Vector{T}) where {T <: Real} 
    lags[findfirst(acf .<= 0.5)]
end

function acw50(lags::Vector{T}, acf::AbstractArray{T}; dims::Int=ndims(acf)) where {T <: Real}
    f = x -> acw50(lags, vec(x))
    return dropdims(mapslices(f, acf, dims=dims), dims=dims)
end

"""
    acw0(lags::Vector{T}, acf::AbstractArray{T}; dims::Int=ndims(acf)) where {T <: Real}

Compute the ACW0 (autocorrelation width at 0) along specified dimension.

# Arguments
- `lags`: Vector of lag values
- `acf`: Array of autocorrelation values
- `dims`: Dimension along which to compute ACW0 (defaults to last dimension)

# Returns
Array with ACW0 values, reduced along specified dimension
"""
function acw0(lags::Vector{T}, acf::Vector{T}) where {T <: Real}
    lags[findfirst(acf .<= 0.0)]
end

function acw0(lags::Vector{T}, acf::AbstractArray{T}; dims::Int=ndims(acf)) where {T <: Real}
    f = x -> acw0(lags, vec(x))
    return dropdims(mapslices(f, acf, dims=dims), dims=dims)
end

function lorentzian(f, u)
    return u[1] ./ (1 .+ (f ./ u[2]) .^ 2)
end

# Define the residual function for NonlinearLeastSquares
function residual_lorentzian!(du, u, p)
    du .= mean(sqrt.(abs2.(lorentzian(p[1], u) .- p[2])))
    return nothing
end

function lorentzian_initial_guess(psd::AbstractVector{<:Real}, freqs::AbstractVector{<:Real};
                                  min_freq::Real=freqs[1],
                                  max_freq::Real=freqs[end])

    # Initial parameter guess
    # u[1]: estimate amplitude from low frequency power
    # u[2]: rough estimate of knee frequency from power spectrum
    initial_amp = mean(psd[freqs.<=min_freq*2])
    half_power = initial_amp / 2
    knee_guess_idx = findlast(psd .>= half_power)

    # Ensure valid initial guesses
    if isnothing(knee_guess_idx)
        knee_guess = (minimum(freqs) + maximum(freqs)) / 2
    else
        knee_guess = freqs[knee_guess_idx]
    end

    u0 = [initial_amp, knee_guess]

    return u0
end

"""
    find_knee_frequency(psd::AbstractArray{T}, freqs::Vector{T}; dims::Int=ndims(psd)) where {T <: Real}

Find the knee frequency by fitting a Lorentzian function to the PSD along specified dimension.

# Arguments
- `psd`: Array of PSD values
- `freqs`: Vector of frequencies
- `dims`: Dimension along which to compute knee frequency (defaults to last dimension)
- `min_freq`: Minimum frequency to consider
- `max_freq`: Maximum frequency to consider

# Returns
Array with knee frequency values, reduced along specified dimension
"""
function find_knee_frequency(psd::Vector{T}, freqs::Vector{T};
                           min_freq::T=freqs[1],
                           max_freq::T=freqs[end]) where {T <: Real}
    # Initial parameter guess
    u0 = lorentzian_initial_guess(psd, freqs, min_freq=min_freq, max_freq=max_freq)
    
    # Set up and solve the nonlinear least squares problem
    prob = NonlinearLeastSquaresProblem(NonlinearFunction(residual_lorentzian!,
                                                         resid_prototype=zeros(2)), u0,
                                       p=[freqs, psd])
    
    sol = NonlinearSolve.solve(prob, FastShortcutNLLSPolyalg(), abstol=5, verbose=false)
    return (sol.u[1], sol.u[2])
end

function find_knee_frequency(psd::AbstractArray{T}, freqs::Vector{T}; 
                           dims::Int=ndims(psd),
                           min_freq::T=freqs[1],
                           max_freq::T=freqs[end]) where {T <: Real}
    f = x -> find_knee_frequency(vec(x), freqs, min_freq=min_freq, max_freq=max_freq)
    dropdims(mapslices(f, psd, dims=dims), dims=dims)
end

"""
    fooof_fit(psd::AbstractArray{T}, freqs::Vector{T}; dims::Int=ndims(psd)) where {T <: Real}

FOOOF style fitting along specified dimension.
1) Fit a Lorentzian to the PSD
2) Subtract Lorentzian
3) Find oscillation peaks
In FOOOF, the following steps are also performed:
4) Fit Gaussian to peaks
5) Iterate until convergence

We only implement the first three steps since our interest is mainly in the knee frequency.

# Arguments
- `psd`: Array of PSD values
- `freqs`: Vector of frequencies
- `dims`: Dimension along which to compute fit (defaults to last dimension)
- `min_freq`: Minimum frequency to consider
- `max_freq`: Maximum frequency to consider
- `oscillation_peak`: Whether to compute oscillation peak

# Returns
Array with fitted parameters, reduced along specified dimension
"""
function fooof_fit(psd::AbstractVector{T}, freqs::AbstractVector{T};
                  min_freq::T=freqs[1],
                  max_freq::T=freqs[end],
                  oscillation_peak::Bool=true) where {T <: Real}
    freq_mask = (freqs .>= min_freq) .& (freqs .<= max_freq)
    fit_psd = psd[freq_mask]
    fit_freqs = freqs[freq_mask]

    # 1) Fit a Lorentzian to the PSD
    amp, knee = find_knee_frequency(fit_psd, fit_freqs; 
                                  min_freq=min_freq, max_freq=max_freq)

    # Return early if oscillation peak not requested
    if !oscillation_peak
        return knee
    end

    # 2) Subtract Lorentzian and find peak
    lorentzian_psd = lorentzian(fit_freqs, [amp, knee])
    residual_psd = fit_psd .- lorentzian_psd
    osc_peak = find_oscillation_peak(residual_psd, fit_freqs; 
                                   min_freq=min_freq, max_freq=max_freq)
    return knee, osc_peak
end

function fooof_fit(psd::AbstractArray{T}, freqs::Vector{T}; 
                  dims::Int=ndims(psd),
                  min_freq::T=freqs[1],
                  max_freq::T=freqs[end],
                  oscillation_peak::Bool=true) where {T <: Real}
    f = x -> fooof_fit(vec(x), freqs, 
                      min_freq=min_freq, max_freq=max_freq,
                      oscillation_peak=oscillation_peak)
    dropdims(mapslices(f, psd, dims=dims), dims=dims)
end

"""
Find the dominant oscillatory peak in the PSD using prominence
"""
function find_oscillation_peak(psd::AbstractVector{<:Real}, freqs::AbstractVector{<:Real};
                               min_freq::Real=5.0 / 1000.0,
                               max_freq::Real=50.0 / 1000.0,
                               min_prominence_ratio::Real=0.1)  # minimum prominence as fraction of max PSD
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
    valid_peaks = peak_indices[prominences.>=min_prominence]
    valid_prominences = prominences[prominences.>=min_prominence]

    # If no peaks meet prominence criterion, return NaN
    if isempty(valid_peaks)
        return NaN
    end

    # Return frequency of the most prominent peak
    best_peak_idx = valid_peaks[argmax(valid_prominences)]
    return search_freqs[best_peak_idx]
end

end # module