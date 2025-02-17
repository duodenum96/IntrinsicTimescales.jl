module OptimizationTau

using Optimization
using Optim
using ForwardDiff
using IntrinsicTimescales

# This thing is not needed.  

function get_u0(model::OneTimescaleModel, sim_summary::AbstractArray{<:Real})
    if model.summary_method == :acf
        tau = fit_expdecay(model.lags_freqs, sim_summary)
    elseif model.summary_method == :psd
        tau = tau_from_knee(find_knee_frequency(sim_summary, model.lags_freqs)[2])
    end
    return [tau]
end

function get_u0(model::OneTimescaleWithMissingModel, sim_summary::AbstractArray{<:Real})
    if model.summary_method == :acf
        tau = fit_expdecay(model.lags_freqs, sim_summary)
    elseif model.summary_method == :psd
        tau = tau_from_knee(find_knee_frequency(sim_summary, model.lags_freqs)[2])
    end
    return [tau]
end

function get_u0(model::OneTimescaleAndOscModel, sim_summary::AbstractArray{<:Real})
    if model.summary_method == :psd
        u0 = lorentzian_initial_guess(sim_summary, model.lags_freqs)
        amp, knee = u0
        lorentzian_psd = lorentzian(model.lags_freqs, [amp, knee])
        residual_psd = sim_summary .- lorentzian_psd

        # Find oscillation peaks
        osc_peak = find_oscillation_peak(residual_psd, model.lags_freqs;
                                         min_freq=model.lags_freqs[1],
                                         max_freq=model.lags_freqs[end])

        return [tau_from_knee(knee), osc_peak, 0.5]
    elseif model.summary_method == :acf
        tau = fit_expdecay(model.lags_freqs, sim_summary)
        u0 = lorentzian_initial_guess(sim_summary, model.lags_freqs)
        amp, knee = u0
        lorentzian_psd = lorentzian(model.lags_freqs, [amp, knee])
        residual_psd = sim_summary .- lorentzian_psd

        # Find oscillation peaks
        osc_peak = find_oscillation_peak(residual_psd, model.lags_freqs;
                                         min_freq=model.lags_freqs[1],
                                         max_freq=model.lags_freqs[end])
        
        return [tau, osc_peak, 0.5]
    end
end

function get_u0(model::OneTimescaleAndOscWithMissingModel, sim_summary::AbstractArray{<:Real})
    if model.summary_method == :psd
        u0 = lorentzian_initial_guess(sim_summary, model.lags_freqs)
        amp, knee = u0
        lorentzian_psd = lorentzian(model.lags_freqs, [amp, knee])
        residual_psd = sim_summary .- lorentzian_psd

        # Find oscillation peaks
        osc_peak = find_oscillation_peak(residual_psd, model.lags_freqs;
                                         min_freq=model.lags_freqs[1],
                                         max_freq=model.lags_freqs[end])

        return [tau_from_knee(knee), osc_peak, 0.5]
    elseif model.summary_method == :acf
        tau = fit_expdecay(model.lags_freqs, sim_summary)
        u0 = lorentzian_initial_guess(sim_summary, model.lags_freqs)
        amp, knee = u0
        lorentzian_psd = lorentzian(model.lags_freqs, [amp, knee])
        residual_psd = sim_summary .- lorentzian_psd

        # Find oscillation peaks
        osc_peak = find_oscillation_peak(residual_psd, model.lags_freqs;
                                         min_freq=model.lags_freqs[1],
                                         max_freq=model.lags_freqs[end])
        
        return [tau, osc_peak, 0.5]
    end
end

end