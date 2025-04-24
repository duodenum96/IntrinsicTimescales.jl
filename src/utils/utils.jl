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
using Logging
using Romberg

export expdecayfit, find_oscillation_peak, find_knee_frequency, fooof_fit,
       lorentzian_initial_guess, lorentzian, expdecay, residual_expdecay!, fit_expdecay,
       acw50, acw50_analytical, acw0, acweuler, tau_from_acw50, tau_from_knee,
       knee_from_tau, acw_romberg

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

function fit_expdecay(lags::AbstractVector{T}, acf::AbstractArray{T};
                      dims::Int=ndims(acf)) where {T <: Real}
    f = x -> fit_expdecay(lags, vec(x))
    return dropdims(mapslices(f, acf, dims=dims), dims=dims)
end

acw50_analytical(tau) = -tau * log(0.5)
tau_from_acw50(acw50) = -acw50 / log(0.5)
tau_from_knee(knee) = 1.0 ./ (2.0 .* pi .* knee)
knee_from_tau(tau) = 1.0 ./ (2.0 .* pi .* tau)

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
function acw50(lags::AbstractVector{T}, acf::AbstractVector{T};
               dims::Int=ndims(acf)) where {T <: Real}
    lags[findfirst(acf .<= 0.5)]
end

function acw50(lags::AbstractVector{T}, acf::AbstractArray{T};
               dims::Int=ndims(acf)) where {T <: Real}
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
    if any(acf .<= 0.0)
        return lags[findfirst(acf .<= 0.0)]
    else
        @warn "No zero crossings found in ACF. Returning NaN."
        return NaN
    end
end

function acw0(lags::AbstractVector{T}, acf::AbstractArray{S};
              dims::Int=ndims(acf)) where {T <: Real, S <: Real}
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
function acweuler(lags::AbstractVector{T},
                  acf::AbstractVector{S}) where {T <: Real, S <: Real}
    lags[findfirst(acf .<= 1 / ℯ)]
end

function acweuler(lags::AbstractVector{T}, acf::AbstractArray{S};
                  dims::Int=ndims(acf)) where {T <: Real, S <: Real}
    f = x -> acweuler(lags, vec(x))
    return dropdims(mapslices(f, acf, dims=dims), dims=dims)
end

"""
    acw_romberg(lags, acf)

Calculate the area under the curve of ACF using Romberg integration.

# Arguments
- `dt::Real`: Time step
- `acf::AbstractVector`: Array of autocorrelation values

# Returns
- AUC of ACF

# Notes
- Returns only the integral value, discarding the error estimate
"""
function acw_romberg(dt::Real, acf::AbstractVector{S}) where {S <: Real}
    return romberg(dt, acf)[1]
end

function acw_romberg(dt::Real, acf::AbstractArray{S};
                     dims::Int=ndims(acf)) where {S <: Real}
    f = x -> acw_romberg(dt, vec(x))
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

"""
    lorentzian_with_exponent(f, u)

Compute Lorentzian function values that allow variable exponent (PLE).

# Arguments
- `f::AbstractVector`: Frequency values
- `u::Vector`: Parameters [amplitude, knee_frequency, exponent]

# Returns
- Vector of Lorentzian values: amp/(1 + (f/knee)²)
"""
function lorentzian_with_exponent(f, u)
    return u[1] ./ (1 .+ ( (f ./ u[2]) .^ u[3] ) )
end

# Define the residual function for NonlinearLeastSquares
function residual_lorentzian!(du, u, p)
    du .= mean(sqrt.(abs2.(lorentzian(p[1], u) .- p[2])))
    return nothing
end

function residual_lorentzian_with_exponent!(du, u, p)
    du .= mean(sqrt.(abs2.(lorentzian_with_exponent(p[1], u) .- p[2])))
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
- Estimates amplitude from average power of low frequencies. 
- Estimates knee frequency from half-power point. 
- If allow_variable_exponent=true, sets initial guess for exponent to 2.0. 
- Used as starting point for nonlinear fitting
"""
function lorentzian_initial_guess(psd::AbstractVector{<:Real},
                                  freqs::AbstractVector{<:Real}; min_freq::Real=freqs[1],
                                  max_freq::Real=freqs[end], 
                                  allow_variable_exponent::Bool=false)
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
    if allow_variable_exponent
        u0 = [initial_amp, knee_guess, 2.0]
    else
        u0 = [initial_amp, knee_guess]
    end

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
# Keyword arguments
- `allow_variable_exponent::Bool=false`: Whether to allow variable exponent (PLE)

# Returns
- Vector of the fit  for the equation amp/(1 + (f/knee)^{exponent}). 
If allow_variable_exponent=false, assumes exponent=2 and returns [amplitude, knee_frequency]. If true, 
returns [amplitude, knee_frequency, exponent].

# Notes
- Uses Lorentzian fitting with NonlinearSolve.jl
- Initial guess for amplitude is based average value of low frequency power. For knee, this is half-power point. For exponent, it is 2.0.  
"""
function find_knee_frequency(psd::AbstractVector{T}, freqs::AbstractVector{T};
                             min_freq::T=freqs[1],
                             max_freq::T=freqs[end],
                             allow_variable_exponent::Bool=false) where {T <: Real}
    # Initial parameter guess
    u0 = lorentzian_initial_guess(psd, freqs; min_freq=min_freq, max_freq=max_freq,
                                  allow_variable_exponent=allow_variable_exponent)

    # Set up and solve the nonlinear least squares problem
    if allow_variable_exponent
        prob = NonlinearLeastSquaresProblem(NonlinearFunction(residual_lorentzian_with_exponent!,
                                                              resid_prototype=zeros(3)), u0,
                                            p=[freqs, psd])
    else
        prob = NonlinearLeastSquaresProblem(NonlinearFunction(residual_lorentzian!,
                                                              resid_prototype=zeros(2)), u0,
                                            p=[freqs, psd])
    end

    sol = NonlinearSolve.solve(prob, FastShortcutNLLSPolyalg(), abstol=5, verbose=false)
    return sol.u
end

function find_knee_frequency(psd::AbstractArray{T}, freqs::AbstractVector{T};
                             dims::Int=ndims(psd),
                             min_freq::T=freqs[1],
                             max_freq::T=freqs[end],
                             allow_variable_exponent::Bool=false) where {T <: Real}
    f = x -> find_knee_frequency(vec(x), freqs; min_freq=min_freq, max_freq=max_freq,
                                 allow_variable_exponent=allow_variable_exponent)
    return dropdims(mapslices(f, psd, dims=dims), dims=dims)
end

"""
    fooof_fit(psd, freqs; dims=ndims(psd), min_freq=freqs[1], max_freq=freqs[end], 
              oscillation_peak=true, max_peaks=3)

Perform FOOOF-style fitting of power spectral density. The default behavior is to fit a Lorentzian with PLE = 2. 
If allow_variable_exponent=true, the function will fit a Lorentzian with variable PLE. 

# Arguments
- `psd::AbstractArray{T}`: Power spectral density values
- `freqs::Vector{T}`: Frequency values
- `dims::Int=ndims(psd)`: Dimension along which to compute
- `min_freq::T=freqs[1]`: Minimum frequency to consider
- `max_freq::T=freqs[end]`: Maximum frequency to consider
- `oscillation_peak::Bool=true`: Whether to compute oscillation peaks
- `max_peaks::Int=3`: Maximum number of oscillatory peaks to fit
- `allow_variable_exponent::Bool=false`: Whether to allow variable exponent (PLE)

# Returns
If return_only_knee=false:
- Tuple of (knee_frequency, oscillation_parameters)
  where oscillation_parameters is Vector of (center_freq, amplitude, std_dev) for each peak
If return_only_knee=true:
- knee_frequency only

# Notes
- Implements iterative FOOOF-style fitting:
  1. Fit initial Lorentzian to PSD
  2. Find and fit Gaussian peaks iteratively
  3. Subtract all Gaussians from original PSD
  4. Refit Lorentzian to cleaned PSD
"""
function fooof_fit(psd::AbstractVector{T}, freqs::AbstractVector{T};
                   min_freq::T=freqs[1],
                   max_freq::T=freqs[end],
                   oscillation_peak::Bool=true,
                   max_peaks::Int=3,
                   return_only_knee::Bool=false,
                   allow_variable_exponent::Bool=false) where {T <: Real}
    freq_mask = (freqs .>= min_freq) .& (freqs .<= max_freq)
    fit_psd = psd[freq_mask]
    fit_freqs = freqs[freq_mask]

    # 1) Initial Lorentzian fit
    fitted_parameters = find_knee_frequency(fit_psd, fit_freqs;
                                    min_freq=min_freq, max_freq=max_freq,
                                    allow_variable_exponent=allow_variable_exponent)

    # Return early if oscillation peak not requested
    if !oscillation_peak
        return fitted_parameters[2]
    end

    # 2) Iteratively find and fit Gaussian peaks
    residual_psd = copy(fit_psd)
    if allow_variable_exponent
        lorentzian_psd = lorentzian_with_exponent(fit_freqs, fitted_parameters)
    else
        lorentzian_psd = lorentzian(fit_freqs, fitted_parameters)
    end

    residual_psd .-= lorentzian_psd

    peaks = []
    for _ in 1:max_peaks
        peak_freq = find_oscillation_peak(residual_psd, fit_freqs;
                                          min_freq=min_freq, max_freq=max_freq)

        isnan(peak_freq) && break

        # Fit Gaussian to the peak
        gaussian_params = fit_gaussian(residual_psd, fit_freqs, peak_freq;
                                       min_freq=min_freq, max_freq=max_freq)

        push!(peaks, gaussian_params)

        # Subtract fitted Gaussian from residual
        residual_psd .-= gaussian(fit_freqs, gaussian_params)
    end

    if isempty(peaks)
        return knee, Vector{Tuple{T, T, T}}()
    end

    # 3) Subtract all Gaussians from original PSD and refit Lorentzian
    cleaned_psd = copy(fit_psd)
    for peak_params in peaks
        cleaned_psd .-= gaussian(fit_freqs, peak_params)
    end

    # 4) Final Lorentzian fit on cleaned PSD
    final_fitted_parameters = find_knee_frequency(cleaned_psd, fit_freqs;
                                                min_freq=min_freq, max_freq=max_freq,
                                                allow_variable_exponent=allow_variable_exponent)

    # Return final knee frequency and all peak parameters
    peak_params = [(p[2], p[1], p[3]) for p in peaks]  # center_freq, amplitude, std_dev
    if return_only_knee
        return final_fitted_parameters[2]
    else
        return final_fitted_parameters[2], peak_params
    end
end

function fooof_fit(psd::AbstractArray{T}, freqs::AbstractVector{T};
                   dims::Int=ndims(psd),
                   min_freq::T=freqs[1],
                   max_freq::T=freqs[end],
                   oscillation_peak::Bool=true,
                   max_peaks::Int=3,
                   return_only_knee::Bool=false,
                   allow_variable_exponent::Bool=false) where {T <: Real}
    f = x -> fooof_fit(vec(x), freqs,
                       min_freq=min_freq, max_freq=max_freq,
                       oscillation_peak=oscillation_peak,
                       max_peaks=max_peaks, return_only_knee=return_only_knee,
                       allow_variable_exponent=allow_variable_exponent)
    return dropdims(mapslices(f, psd, dims=dims), dims=dims)
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

"""
    gaussian(f, u)

Gaussian function for fitting oscillations. 

# Arguments
- `f::AbstractVector`: Frequency values
- `u::Vector`: Parameters [amplitude, center_freq, std_dev]

# Returns
- Vector of Gaussian values: amp * exp(-(f-center)²/(2*std²))
"""
function gaussian(f, u)
    return u[1] .* exp.(-(f .- u[2]) .^ 2 ./ (2 * u[3]^2))
end

# Define the residual function for NonlinearSolve
function residual_gaussian!(du, u, p)
    du .= mean(sqrt.(abs2.(gaussian(p[1], u) .- p[2])))
    return nothing
end

"""
    fit_gaussian(psd, freqs, initial_peak; min_freq=freqs[1], max_freq=freqs[end])

Fit Gaussian to power spectral density around a peak.

# Arguments
- `psd::AbstractVector{<:Real}`: Power spectral density values
- `freqs::AbstractVector{<:Real}`: Frequency values
- `initial_peak::Real`: Initial guess for center frequency
- `min_freq::Real`: Minimum frequency to consider
- `max_freq::Real`: Maximum frequency to consider

# Returns
- Vector{Float64}: Fitted parameters [amplitude, center_freq, std_dev]

# Notes
- Uses initial peak location from find_oscillation_peak
"""
function fit_gaussian(psd::AbstractVector{<:Real}, freqs::AbstractVector{<:Real},
                      initial_peak::Real;
                      min_freq::Real=freqs[1], max_freq::Real=freqs[end])
    # Find peak amplitude and rough width estimate
    peak_idx = argmin(abs.(freqs .- initial_peak))
    initial_amp = psd[peak_idx]

    # Estimate width as distance to half maximum
    half_max = initial_amp / 2
    left_idx = findlast(psd[1:peak_idx] .<= half_max)
    right_idx = findfirst(psd[peak_idx:end] .<= half_max)

    if isnothing(left_idx) || isnothing(right_idx)
        initial_std = (max_freq - min_freq) / 10  # fallback width estimate
    else
        right_idx = peak_idx + right_idx - 1
        width = freqs[right_idx] - freqs[left_idx]
        initial_std = width / 2.355  # convert FWHM to standard deviation
    end

    u0 = [initial_amp, initial_peak, initial_std]

    # Set up and solve the nonlinear least squares problem
    prob = NonlinearLeastSquaresProblem(NonlinearFunction(residual_gaussian!,
                                                          resid_prototype=zeros(3)), u0,
                                        p=[freqs, psd])

    sol = NonlinearSolve.solve(prob, FastShortcutNLLSPolyalg(), abstol=5, verbose=false)
    return sol.u
end

end # module