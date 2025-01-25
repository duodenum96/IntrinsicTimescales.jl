# src/models/one_timescale.jl

module OneTimescale

using Distributions
using ..Models
using ..OrnsteinUhlenbeck
using BayesianINT
using Optimization
using OptimizationOptimJL

export one_timescale_model, OneTimescaleModel

function informed_prior(data_sum_stats::Vector{<:Real}, lags_freqs; summary_method=:acf)
    if summary_method == :acf
        tau = fit_expdecay(lags_freqs, data_sum_stats)
        return [Normal(tau, 20.0)] # Convert to ms
    elseif summary_method == :psd
        tau = tau_from_knee(find_knee_frequency(data_sum_stats, lags_freqs)[2]) # Get knee frequency from Lorentzian fit
        return [Normal(tau, 20.0)] # Convert to ms
    end
end

"""
One-timescale OU process model
"""
struct OneTimescaleModel <: AbstractTimescaleModel
    data::AbstractArray{<:Real}
    time::AbstractVector{<:Real}
    fit_method::Symbol # can be "abc", "optimization", "acw"
    summary_method::Symbol # :psd or :acf
    lags_freqs::Union{Real, AbstractVector} # lags if summary method is acf, freqs otherwise, If the user enters an empty vector, will use defaults. 
    prior::Union{Vector{<:Distribution}, Distribution, String, Nothing} # Vector of prior distributions, single distribution, or string for "informed_prior"
    optalg # Optimization algorithm for Optimization.jl
    acwtypes::Union{Vector{<:Symbol}, Symbol, Nothing} # Types of ACW: ACW-50, ACW-0, ACW-euler, tau, knee frequency
    distance_method::Symbol # :linear or :logarithmic
    data_sum_stats::AbstractArray{<:Real}
    dt::Real
    T::Real
    numTrials::Real
    data_mean::Real
    data_sd::Real
    freqlims::Union{Tuple{Real, Real}, Nothing}
    n_lags::Union{Int, Nothing}
    freq_idx::Union{Vector{Bool}, Nothing}
    dims::Int
    distance_combined::Bool
    weights::Vector{Real}
    data_tau::Union{Real, Nothing}
    u0::Union{Vector{Real}, Nothing} # Initial guess for optimization
end

"""
5 ways to construct a OneTimescaleModel:
1 - summary_method == :acf, fitmethod == :abc
one_timescale_model(data, time, :abc; summary_method=:acf, prior=nothing, n_lags=nothing, 
                    distance_method=nothing, distance_combined=false, weights=nothing)

2 - summary_method == :acf, fitmethod == :optimization
one_timescale_model(data, time, :optimization; summary_method=:acf, n_lags=nothing, 
                    optalg=nothing, distance_method=nothing, distance_combined=false, weights=nothing, u0=nothing)

3 - summary_method == :psd, fitmethod == :abc
one_timescale_model(data, time, :abc, summary_method == :psd, prior=nothing, 
                    distance_method=nothing, freqlims=nothing, distance_combined=false, weights=nothing)

4 - summary_method == :psd, fitmethod == :optimization
one_timescale_model(data, time, :optimization, summary_method=:psd, optalg=nothing, 
                    distance_method=nothing, distance_combined=false, weights=nothing, u0=nothing)

5 - summary_method == nothing, fitmethod == :acw
one_timescale_model(data, time, :acw; summary_method=nothing, n_lags=nothing, 
                    acwtypes=[:acw0, acw50, acwe, tau, knee], dims=ndims(data))
"""
function one_timescale_model(data, time, fit_method; summary_method=:acf,
                             data_sum_stats=nothing,
                             lags_freqs=nothing, prior=nothing, n_lags=nothing,
                             optalg=nothing, acwtypes=nothing, distance_method=nothing,
                             dt=time[2] - time[1], T=time[end], numTrials=size(data, 1),
                             data_mean=mean(data),
                             data_sd=std(data), freqlims=nothing, freq_idx=nothing,
                             dims=ndims(data), distance_combined=false,
                             weights=[0.5, 0.5], data_tau=nothing, u0=nothing)

    # case 1: acf and abc or optim
    if summary_method == :acf && (fit_method == :abc || fit_method == :optimization)
        acf = comp_ac_fft(data)
        acf_mean = mean(acf, dims=1)[:]
        lags_samples = 0.0:(size(data, dims)-1)
        if isnothing(n_lags)
            n_lags = floor(Int, acw0(lags_samples, acf_mean) * 1.5)
        end
        lags_freqs = collect(lags_samples * dt)[1:n_lags]
        data_sum_stats = acf_mean[1:n_lags]
        if (isnothing(prior) || prior == "informed_prior") && fit_method == :abc
            prior = informed_prior(data_sum_stats, lags_freqs;
                                   summary_method=summary_method) # TODO: We are calculating a bunch of stuff twice here :(
        end
        if isnothing(distance_method)
            distance_method = :linear
        end

        if distance_combined || fit_method == :optimization
            data_tau = fit_expdecay(lags_freqs, data_sum_stats)
            u0 = [data_tau]
        end


        return OneTimescaleModel(data, time, fit_method, summary_method, lags_freqs, prior,
                                 optalg, acwtypes, distance_method, data_sum_stats, dt, T,
                                 numTrials, data_mean, data_sd, freqlims, n_lags, freq_idx,
                                 dims, distance_combined, weights, data_tau, u0)
        # case 2: psd and abc or optim
    elseif summary_method == :psd && (fit_method == :abc || fit_method == :optimization)
        fs = 1 / dt
        psd, freqs = comp_psd(data, fs)
        mean_psd = mean(psd, dims=1)
        if isnothing(freqlims)
            freqlims = (0.5 / 1000.0, 100.0 / 1000.0) # Convert to kHz (units in ms)
        end
        freq_idx = (freqs .< freqlims[2]) .&& (freqs .> freqlims[1])
        lags_freqs = freqs[freq_idx]
        data_sum_stats = mean_psd[freq_idx]
        if (isnothing(prior) || prior == "informed_prior") && fit_method == :abc
            prior = informed_prior(data_sum_stats, lags_freqs;
                                   summary_method=summary_method)
        end
        if isnothing(distance_method)
            distance_method = :logarithmic
        end

        if distance_combined || fit_method == :optimization
            data_tau = tau_from_knee(find_knee_frequency(data_sum_stats, lags_freqs)[2])
            u0 = [data_tau]
        end

        if isnothing(optalg)
            optalg = Optim.LBFGS()
        end 

        return OneTimescaleModel(data, time, fit_method, summary_method, lags_freqs, prior,
                                 optalg, acwtypes, distance_method, data_sum_stats, dt, T,
                                 numTrials, data_mean, data_sd, freqlims, n_lags, freq_idx,
                                 dims, distance_combined, weights, data_tau, u0)
        # case 3: acw
    elseif fit_method == :acw
        possible_acwtypes = [:acw0, :acw50, :acweuler, :tau, :knee]
        acf_acwtypes = [:acw0, :acw50, :acweuler, :tau]
        n_acw = length(acwtypes)
        if n_acw == 0
            error("No ACW types specified. Possible ACW types: $(possible_acwtypes)")
        end
        result = Vector{Vector{<:Real}}(undef, n_acw)
        acwtypes = check_acwtypes(acwtypes, possible_acwtypes)
        if any(in.(acf_acwtypes, [acwtypes]))
            acf = comp_ac_fft(data; dims=dims)
            lags_samples = 0.0:(size(data, dims)-1)
            lags = lags_samples * dt
            if any(in.(:acw0, [acwtypes]))
                acw0_idx = findfirst(acwtypes .== :acw0)
                acw0_result = acw0(lags, acf; dims=dims)
                result[acw0_idx] = acw0_result
            end
            if any(in.(:acw50, [acwtypes]))
                acw50_idx = findfirst(acwtypes .== :acw50)
                acw50_result = acw50(lags, acf; dims=dims)
                result[acw50_idx] = acw50_result
            end
            if any(in.(:acweuler, [acwtypes]))
                acweuler_idx = findfirst(acwtypes .== :acweuler)
                acweuler_result = acweuler(lags, acf; dims=dims)
                result[acweuler_idx] = acweuler_result
            end
            if any(in.(:tau, [acwtypes]))
                tau_idx = findfirst(acwtypes .== :tau)
                tau_result = fit_expdecay(collect(lags), acf; dims=dims)
                result[tau_idx] = tau_result
            end
        end

        if any(in.(:knee, [acwtypes]))
            knee_idx = findfirst(acwtypes .== :knee)
            fs = 1 / dt
            psd, freqs = comp_psd(data, fs, dims=dims)
            knee_result = tau_from_knee(find_knee_frequency(psd, freqs; dims=dims))
            result[knee_idx] = knee_result
        end
        return result
    end
end

# Implementation of required methods (theta is the tau)
function Models.generate_data(model::OneTimescaleModel, theta)
    return generate_ou_process(theta[1], model.data_sd, model.dt, model.T, model.numTrials;
                               backend="sciml")
end

function Models.summary_stats(model::OneTimescaleModel, data)
    if model.summary_method == :acf
        return mean(comp_ac_fft(data; n_lags=model.n_lags), dims=1)[:]
    elseif model.summary_method == :psd
        return mean(comp_psd(data, 1 / model.dt)[1], dims=1)[:][model.freq_idx]
    else
        throw(ArgumentError("Summary method must be :acf or :psd"))
    end
end

"""
Combined distance as a linear combination of L2 distance between simulation_summary and data_summary and 
L2 distance between fitted timescale values between them. 
weights: weights for the two distances respectively
"""
function combined_distance(model::OneTimescaleModel, simulation_summary, data_summary,
                           weights,
                           data_tau, simulation_tau)
    if model.distance_method == :linear
        distance_1 = linear_distance(simulation_summary, data_summary)
    elseif model.distance_method == :logarithmic
        distance_1 = logarithmic_distance(simulation_summary, data_summary)
    else
        throw(ArgumentError("Distance method must be :linear or :logarithmic"))
    end
    distance_2 = linear_distance(data_tau, simulation_tau)
    return weights[1] * distance_1 + weights[2] * distance_2
end

function Models.distance_function(model::OneTimescaleModel, sum_stats, data_sum_stats)
    if model.distance_combined
        if model.summary_method == :acf
            simulation_tau = fit_expdecay(model.lags_freqs, sum_stats)
        elseif model.summary_method == :psd
            simulation_tau = tau_from_knee(find_knee_frequency(sum_stats, model.lags_freqs)[2])
        end
        return combined_distance(model, sum_stats, data_sum_stats, model.weights,
                                 model.data_tau, simulation_tau)
    elseif model.distance_method == :linear
        return linear_distance(sum_stats, data_sum_stats)
    elseif model.distance_method == :logarithmic
        return logarithmic_distance(sum_stats, data_sum_stats)
    else
        throw(ArgumentError("Distance method must be :linear or :logarithmic"))
    end
end

function Models.solve(model::OneTimescaleModel, param_dict=nothing)
    if model.fit_method == :abc
        if isnothing(param_dict)
            param_dict = get_param_dict_abc()
        end

        abc_record = pmc_abc(model;
                             # Basic ABC parameters
                             epsilon_0=param_dict[:epsilon_0],
                             max_iter=param_dict[:max_iter],
                             min_accepted=param_dict[:min_accepted],
                             steps=param_dict[:steps],
                             sample_only=param_dict[:sample_only],
                             minAccRate=param_dict[:minAccRate],
                             target_acc_rate=param_dict[:target_acc_rate],
                             target_epsilon=param_dict[:target_epsilon],
                             show_progress=param_dict[:show_progress],
                             verbose=param_dict[:verbose],
                             jitter=param_dict[:jitter],
                             cov_scale=param_dict[:cov_scale],
                             distance_max=param_dict[:distance_max],
                             quantile_lower=param_dict[:quantile_lower],
                             quantile_upper=param_dict[:quantile_upper],
                             quantile_init=param_dict[:quantile_init],
                             acc_rate_buffer=param_dict[:acc_rate_buffer],
                             alpha_max=param_dict[:alpha_max],
                             alpha_min=param_dict[:alpha_min],
                             acc_rate_far=param_dict[:acc_rate_far],
                             acc_rate_close=param_dict[:acc_rate_close],
                             alpha_far_mult=param_dict[:alpha_far_mult],
                             alpha_close_mult=param_dict[:alpha_close_mult])
    posterior_samples = abc_record[end].theta_accepted
    posterior_MAP = find_MAP(posterior_samples, param_dict[:N])
    return posterior_samples, posterior_MAP, abc_record

    elseif model.fit_method == :optimization
        u0 = Float64[copy(model.u0)...] # The optimizer should now the type.
        optalg = model.optalg
        function loss_function(u, p)
            # Generate data and compute distance
            sim_data = Models.generate_data(model, u)
            sim_stats = Models.summary_stats(model, sim_data)
            distance = Models.distance_function(model, sim_stats, model.data_sum_stats)
            println(distance)
            return distance
        end
        p = ComponentVector(a = 1.0)
        optf = OptimizationFunction(loss_function, DifferentiationInterface.AutoEnzyme())
        prob = OptimizationProblem(optf, u0, p)
        sol = solve(prob, BFGS())
        return sol
    end
end

end # module OneTimescale 