# src/utils/utils.jl

"""
    Utils

Module providing utility functions for time series analysis, including:
- Exponential decay fitting
- Oscillation peak detection
- Knee frequency estimation
- Lorentzian fitting
- ACF width calculations
"""
module Utils

using Statistics
using NonlinearSolve
export expdecayfit, find_oscillation_peak, find_knee_frequency, fooof_fit,
       lorentzian_initial_guess, lorentzian, expdecay, residual_expdecay!, fit_expdecay,
       acw50, acw50_analytical, acw0, acweuler, tau_from_acw50, tau_from_knee, knee_from_tau

"""
    expdecay(tau, lags)

Compute exponential decay function.

# Arguments
- `tau::Real`: Timescale parameter
- `lags::AbstractVector`: Time lags

# Returns
- Vector of exp(-t/tau) values

# Notes
- Used for fitting autocorrelation functions
- Assumes exponential decay model: acf = exp(-t/tau)
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

"""
    fit_expdecay(lags, acf; dims=ndims(acf))

Fit exponential decay to autocorrelation function.

# Arguments
- `lags::AbstractVector{T}`: Time lags
- `acf::AbstractArray{T}`: Autocorrelation values
- `dims::Int=ndims(acf)`: Dimension along which to fit

# Returns
- Fitted timescale parameter(s)

# Notes
- Uses NonlinearSolve.jl with FastShortcutNLLSPolyalg
- Initial guess based on ACW50
"""
function fit_expdecay(lags::AbstractVector{T}, acf::AbstractVector{T}) where {T <: Real}
    u0 = [tau_from_acw50(acw50(lags, acf))]
    prob = NonlinearLeastSquaresProblem(NonlinearFunction(residual_expdecay!,
                                                          resid_prototype=zeros(1)), u0,
                                        p=[lags, acf])
    sol = NonlinearSolve.solve(prob, FastShortcutNLLSPolyalg(), reltol=0.001, verbose=false) # TODO: Find a reasonable tolerance. 
    return sol.u[1]
end

function fit_expdecay(lags::AbstractVector{T}, acf::AbstractArray{T}; dims::Int=ndims(acf)) where {T <: Real}
    f = x -> fit_expdecay(lags, vec(x))
    return dropdims(mapslices(f, acf, dims=dims), dims=dims)
end

acw50_analytical(tau) = -tau * log(0.5)
tau_from_acw50(acw50) = -acw50 / log(0.5)
tau_from_knee(knee) = 1 ./ (2 .* pi .* knee)
knee_from_tau(tau) = 1 ./ (2 .* pi .* tau)

"""
    acw50(lags, acf; dims=ndims(acf))

Compute the ACW50 (autocorrelation width at 50%) along specified dimension.

# Arguments
- `lags::AbstractVector{T}`: Vector of lag values
- `acf::AbstractArray{T}`: Array of autocorrelation values
- `dims::Int=ndims(acf)`: Dimension along which to compute ACW50

# Returns
- First lag where autocorrelation falls below 0.5

# Notes
- Used for estimating characteristic timescales
- Related to tau by: tau = -acw50/log(0.5)
"""
function acw50(lags::AbstractVector{T}, acf::AbstractVector{T}; dims::Int=ndims(acf)) where {T <: Real}
    lags[findfirst(acf .<= 0.5)]
end

function acw50(lags::AbstractVector{T}, acf::AbstractArray{T}; dims::Int=ndims(acf)) where {T <: Real}
    f = x -> acw50(lags, vec(x))
    return dropdims(mapslices(f, acf, dims=dims), dims=dims)
end

"""
    acw0(lags, acf; dims=ndims(acf))

Compute the ACW0 (autocorrelation width at zero crossing) along specified dimension.

# Arguments
- `lags::AbstractVector{T}`: Vector of lag values
- `acf::AbstractArray{T}`: Array of autocorrelation values
- `dims::Int=ndims(acf)`: Dimension along which to compute ACW0

# Returns
- First lag where autocorrelation crosses zero

# Notes
- Alternative measure of characteristic timescale
- More sensitive to noise than ACW50
"""
function acw0(lags::AbstractVector{T}, acf::AbstractVector{S}) where {T <: Real, S <: Real}
    lags[findfirst(acf .<= 0.0)]
end

function acw0(lags::AbstractVector{T}, acf::AbstractArray{S}; dims::Int=ndims(acf)) where {T <: Real, S <: Real}
    f = x -> acw0(lags, vec(x))
    return dropdims(mapslices(f, acf, dims=dims), dims=dims)
end

"""
    acweuler(lags, acf; dims=ndims(acf))

Compute the ACW at 1/e (≈ 0.368) along specified dimension.

# Arguments
- `lags::AbstractVector{T}`: Vector of lag values
- `acf::AbstractArray{S}`: Array of autocorrelation values
- `dims::Int=ndims(acf)`: Dimension along which to compute

# Returns
- First lag where autocorrelation falls below 1/e

# Notes
- For exponential decay, equals the timescale parameter tau
"""
function acweuler(lags::AbstractVector{T}, acf::AbstractVector{S}) where {T <: Real, S <: Real}
    lags[findfirst(acf .<= 1/ℯ)]
end

function acweuler(lags::AbstractVector{T}, acf::AbstractArray{S}; dims::Int=ndims(acf)) where {T <: Real, S <: Real}
    f = x -> acweuler(lags, vec(x))
    return dropdims(mapslices(f, acf, dims=dims), dims=dims)
end

"""
    lorentzian(f, u)

Compute Lorentzian function values.

# Arguments
- `f::AbstractVector`: Frequency values
- `u::Vector`: Parameters [amplitude, knee_frequency]

# Returns
- Vector of Lorentzian values: amp/(1 + (f/knee)²)
"""
function lorentzian(f, u)
    return u[1] ./ (1 .+ (f ./ u[2]) .^ 2)
end

# Define the residual function for NonlinearLeastSquares
function residual_lorentzian!(du, u, p)
    du .= mean(sqrt.(abs2.(lorentzian(p[1], u) .- p[2])))
    return nothing
end

"""
    lorentzian_initial_guess(psd, freqs; min_freq=freqs[1], max_freq=freqs[end])

Estimate initial parameters for Lorentzian fitting.

# Arguments
- `psd::AbstractVector{<:Real}`: Power spectral density values
- `freqs::AbstractVector{<:Real}`: Frequency values
- `min_freq::Real`: Minimum frequency to consider
- `max_freq::Real`: Maximum frequency to consider

# Returns
- Vector{Float64}: Initial guess for [amplitude, knee_frequency]

# Notes
- Estimates amplitude from maximum PSD value
- Estimates knee frequency from half-power point
- Used as starting point for nonlinear fitting
"""
function lorentzian_initial_guess(psd::AbstractVector{<:Real}, freqs::AbstractVector{<:Real}; kwargs...)
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
    find_knee_frequency(psd, freqs; dims=ndims(psd), min_freq=freqs[1], max_freq=freqs[end])

Find knee frequency by fitting Lorentzian to power spectral density.

# Arguments
- `psd::AbstractArray{T}`: Power spectral density values
- `freqs::Vector{T}`: Frequency values
- `dims::Int=ndims(psd)`: Dimension along which to compute
- `min_freq::T=freqs[1]`: Minimum frequency to consider
- `max_freq::T=freqs[end]`: Maximum frequency to consider

# Returns
- Knee frequency values (frequency at half power)

# Notes
- Uses Lorentzian fitting with NonlinearSolve.jl
- Initial guess based on half-power point
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
    f = x -> find_knee_frequency(vec(x), freqs, min_freq=min_freq, max_freq=max_freq)[2]
    dropdims(mapslices(f, psd, dims=dims), dims=dims)
end

"""
    fooof_fit(psd, freqs; dims=ndims(psd), min_freq=freqs[1], max_freq=freqs[end], oscillation_peak=true)

Perform FOOOF-style fitting of power spectral density.

# Arguments
- `psd::AbstractArray{T}`: Power spectral density values
- `freqs::Vector{T}`: Frequency values
- `dims::Int=ndims(psd)`: Dimension along which to compute
- `min_freq::T=freqs[1]`: Minimum frequency to consider
- `max_freq::T=freqs[end]`: Maximum frequency to consider
- `oscillation_peak::Bool=true`: Whether to compute oscillation peak

# Returns
If oscillation_peak=true:
- Tuple of (knee_frequency, oscillation_peak_frequency)
If oscillation_peak=false:
- knee_frequency only

# Notes
- Implements first 3 steps of FOOOF algorithm:
  1. Fit Lorentzian to PSD
  2. Subtract Lorentzian
  3. Find oscillation peaks
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

function fooof_fit(psd::AbstractArray{T}, freqs::AbstractVector{T}; 
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
    find_oscillation_peak(psd, freqs; min_freq=5.0/1000.0, max_freq=50.0/1000.0, min_prominence_ratio=0.1)

Find dominant oscillatory peak in power spectral density.

# Arguments
- `psd::AbstractVector`: Power spectral density values
- `freqs::AbstractVector`: Frequency values
- `min_freq::Real=5.0/1000.0`: Minimum frequency to consider
- `max_freq::Real=50.0/1000.0`: Maximum frequency to consider
- `min_prominence_ratio::Real=0.1`: Minimum peak prominence as fraction of max PSD

# Returns
- Frequency of most prominent peak, or NaN if no significant peak found

# Notes
- Uses peak prominence for robustness
- Filters peaks by minimum prominence threshold
- Returns NaN if no peaks meet criteria
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