module ABCdeZExt

export get_param_dict_abcdemc, get_param_dict_abcdesmc, 
    abcdez_inference

import IntrinsicTimescales as it
import ABCdeZ as abc
import Distributions as dist
import Random

function it.get_param_dict_abcdesmc()
    return Dict(
        :nparticles => 100,
        :α => 0.95,
        :δess => 0.5,
        :nsims_max => 10^7,
        :Kmcmc => 3,
        :Kmcmc_min => 1.0,
        :ABCk => abc.Indicator0toϵ,
        :facc_min => 0.15,
        :facc_tune=>0.975,
        :verbose=>true,
        :verboseout=>true,
        :rng=>Random.GLOBAL_RNG,
        :parallel=>false
    )
end

function it.get_param_dict_abcdemc()
    return Dict(
        :nparticles => 50,
        :generations => 20,
        :verbose => true,
        :rng=>Random.GLOBAL_RNG,
        :parallel => true
    )
end

function it.abcdez_inference(model, ϵ_target, param_dict, method)
    distance_function!(θ, ve) = it.generate_data_and_reduce(model, θ), nothing
    prior = abc.Factored(model.prior...)
    if method == :abcdemc
        result = _abcdez_inference_mc(ϵ_target, param_dict, distance_function!, prior)
    elseif method == :abcdesmc
        result = _abcdez_inference_smc(ϵ_target, param_dict, distance_function!, prior)
    else
        error("method should be either :abcdemc or :abcdesmc")
    end
    return result
end

"""
Perform ABCdeZ using abcdemc
"""
function _abcdez_inference_mc(ϵ_target, param_dict_mc, distance_function!, prior)
    results = abc.abcdemc(prior, distance_function!, ϵ_target, nothing; param_dict_mc...)
    return results
end

function _abcdez_inference_smc(ϵ_target, param_dict_mc, distance_function!, prior)
    results = abc.abcdesmc(prior, distance_function!, ϵ_target, nothing; param_dict_mc...)
    return results
end

end