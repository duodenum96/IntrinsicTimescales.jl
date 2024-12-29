# src/utils/utils.jl
module Utils
using LsqFit
export expdecayfit
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

end # module