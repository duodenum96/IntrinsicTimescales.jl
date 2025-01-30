module ACW

using INT

export acw, acw_container

struct acw_container
    data::AbstractArray{<:Real}
    fs::Real
    acwtypes::Union{Vector{<:Symbol}, Symbol, Nothing} # Types of ACW: ACW-50, ACW-0, ACW-euler, tau, knee frequency
    n_lags::Union{Int, Nothing}
    freqlims::Union{Tuple{Real, Real}, Nothing}
    acw_results::Vector{<:Real}
end

possible_acwtypes = [:acw0, :acw50, :acweuler, :tau, :knee]

function acw(data, fs, acwtypes=possible_acwtypes, n_lags=nothing, freqlims=nothing,
             dims=size(data, ndims(data)))

    if data isa AbstractVector
        data = reshape(data, (1, length(data)))
        dims = size(data, ndims(data))
    end

    if acwtypes isa Symbol
        acwtypes = [acwtypes]
    end

    dt = 1 / fs
    acf_acwtypes = [:acw0, :acw50, :acweuler, :tau]
    n_acw = length(acwtypes)
    if n_acw == 0
        error("No ACW types specified. Possible ACW types: $(possible_acwtypes)")
    end
    # Check if any requested acwtype is not in possible_acwtypes
    result = Vector{Vector{<:Real}}(undef, n_acw)
    acwtypes = check_acwtypes(acwtypes, possible_acwtypes)

    if isnothing(n_lags)
        n_lags = size(data, dims)
    end

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
