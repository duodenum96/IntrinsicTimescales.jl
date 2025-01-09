# src/utils/utils.jl

module Utils

using Statistics
using NonlinearSolve
export expdecayfit, find_oscillation_peak, find_knee_frequency, fooof_fit, lorentzian_initial_guess, lorentzian

"""
Exponential decay fit
acf = u[1] * exp(-u[2] * lags)
acf is a 1D vector (ACF)
lags is a 1D vector (x axis)
"""
function expdecayfit(acf, lags)
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
    return u[1] ./ (1 .+ (f ./ u[2]) .^ 2)
end

# Define the residual function for NonlinearLeastSquares
function residual_lorentzian!(du, u, p)
    du .= mean(sqrt.(abs2.(lorentzian(p[1], u) .- p[2])))
    return nothing
end

function lorentzian_initial_guess(psd::Vector{Float64}, freqs::Vector{Float64};
                                  min_freq::Float64=freqs[1],
                                  max_freq::Float64=freqs[end])

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
Find the knee frequency by fitting a Lorentzian function to the PSD
Returns the frequency where power drops to half of its peak value
"""
function find_knee_frequency(psd::Vector{Float64}, freqs::Vector{Float64};
                             min_freq::Float64=freqs[1],
                             max_freq::Float64=freqs[end])

    # Initial parameter guess
    u0 = lorentzian_initial_guess(psd, freqs, min_freq=min_freq, max_freq=max_freq)

    # Set up and solve the nonlinear least squares problem
    prob = NonlinearLeastSquaresProblem(NonlinearFunction(residual_lorentzian!,
                                                          resid_prototype=zeros(2)), u0,
                                        p=[freqs, psd])

    # TODO: Find a reasonable tolerance. Even in very good fits, residual error is 4.5. 
    sol = NonlinearSolve.solve(prob, FastShortcutNLLSPolyalg(), abstol=5, verbose=false)
    return (sol.u[1], sol.u[2])
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
                   min_freq::Float64=freqs[1],
                   max_freq::Float64=freqs[end])
    freq_mask = (freqs .>= min_freq) .& (freqs .<= max_freq)
    fit_psd = psd[freq_mask]
    fit_freqs = freqs[freq_mask]

    # 1) Fit a Lorentzian to the PSD
    amp, knee = find_knee_frequency(fit_psd, fit_freqs; min_freq=min_freq,
                                    max_freq=max_freq)

    # 2) Subtract Lorentzian
    lorentzian_psd = lorentzian(fit_freqs, [amp, knee])
    residual_psd = fit_psd .- lorentzian_psd

    # 3) Find oscillation peaks
    osc_peak = find_oscillation_peak(residual_psd, fit_freqs; min_freq=min_freq,
                                     max_freq=max_freq)
    return knee, osc_peak
end

end # module