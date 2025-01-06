# src/utils/utils.jl

module Utils

using Statistics
using NonlinearSolve
export expdecayfit, find_oscillation_peak, find_knee_frequency

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

function lorentzian(f, u)
    return u[1] ./ (1 .+ (f ./ u[2]).^2)
end

# Define the residual function for NonlinearLeastSquares
function residual!(du, u, p)
    du .= mean(sqrt.(abs2.(lorentzian(fit_freqs, u) .- fit_psd)))
    return nothing
end

function lorentzian_initial_guess(psd::Vector{Float64}, freqs::Vector{Float64};
                                  min_freq::Float64=0.1 / 1000.0,
                                  max_freq::Float64=100.0 / 1000.0)
    # Consider only frequencies in the specified range
    freq_mask = (freqs .>= min_freq) .& (freqs .<= max_freq)
    fit_psd = psd[freq_mask]
    fit_freqs = freqs[freq_mask]

    # Initial parameter guess
    # u[1]: estimate amplitude from low frequency power
    # u[2]: rough estimate of knee frequency from power spectrum
    initial_amp = mean(fit_psd[fit_freqs.<=min_freq*2])
    half_power = initial_amp / 2
    knee_guess_idx = findlast(fit_psd .>= half_power)

    # Ensure valid initial guesses
    if isnothing(knee_guess_idx)
        knee_guess = (minimum(fit_freqs) + maximum(fit_freqs)) / 2
    else
        knee_guess = fit_freqs[knee_guess_idx]
    end

    u0 = [initial_amp, knee_guess]

    return u0
end

"""
Find the knee frequency by fitting a Lorentzian function to the PSD
Returns the frequency where power drops to half of its peak value
"""
function find_knee_frequency(psd::Vector{Float64}, freqs::Vector{Float64};
                             min_freq::Float64=0.1 / 1000.0,
                             max_freq::Float64=100.0 / 1000.0)
    # Consider only frequencies in the specified range
    freq_mask = (freqs .>= min_freq) .& (freqs .<= max_freq)
    fit_psd = psd[freq_mask]
    fit_freqs = freqs[freq_mask]

    # Initial parameter guess
    u0 = lorentzian_initial_guess(fit_psd, fit_freqs)

    # Set up and solve the nonlinear least squares problem
    prob = NonlinearLeastSquaresProblem(NonlinearFunction(residual!,
                                                          resid_prototype=zeros(2)), u0)

    try
        # TODO: Find a reasonable tolerance. Even in very good fits, residual error is 4.5. 
        sol = NonlinearSolve.solve(prob, FastShortcutNLLSPolyalg(), abstol=5, verbose=false)
        if SciMLBase.successful_retcode(sol)
            # Return the characteristic frequency (knee frequency)
            return sol.u[2]
        end
    catch e
        @warn "Fitting failed: $e"
    end

    return NaN
end

"""
FOOOF style fitting (Donoghue et al. 2020 Nat. Neuroscience)
1) Fit a Lorentzian to the PSD
2) Subtract Lorentzian
3) Find oscillation peaks
We only implement the first three steps since we aren't interested in bandwidth, prominence etc. In FOOOF, 
the following steps are also performed:
4) Fit Gaussian to peaks
5) Iterate until convergence
"""
function fooof_fit(psd::Vector{Float64}, freqs::Vector{Float64};
                   min_freq::Float64=0.1 / 1000.0,
                   max_freq::Float64=100.0 / 1000.0)

    freq_mask = (freqs .>= min_freq) .& (freqs .<= max_freq)
    fit_psd = psd[freq_mask]
    fit_freqs = freqs[freq_mask]

    # 1) Fit a Lorentzian to the PSD


end # module


end # module