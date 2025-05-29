module Plotting

export acwplot, posterior_predictive

"""
    acwplot(acwresults::ACWResults; only_acf::Bool=false, only_psd::Bool=false, show::Bool=true)

Placeholder function for plotting autocorrelation function (ACF) and power spectral density (PSD) results.

This function is implemented in the `IntPlottingExt` extension package, which is automatically
loaded when `Plots.jl` is available. The extension provides comprehensive visualization
capabilities for ACW analysis results.

# Functionality (available when Plots.jl is loaded):
- Plots autocorrelation function and/or power spectral density from `ACWResults`
- Supports plotting either ACF alone, PSD alone, or both in a subplot layout
- ACF plots show individual traces in color with mean overlaid in black
- PSD plots use logarithmic scales for both axes
- Handles up to 2-dimensional ACF data

# Arguments (when extension is loaded):
- `acwresults::ACWResults`: Container with ACF and/or PSD analysis results
- `only_acf::Bool=false`: If true, plot only the autocorrelation function
- `only_psd::Bool=false`: If true, plot only the power spectral density  
- `show::Bool=true`: Whether to display the plot immediately

# Requirements:
- Requires `Plots.jl` to be loaded for the extension to activate
- Install with: `using Plots` or `import Plots`

See `IntPlottingExt` documentation for complete details.
"""
function acwplot end

"""
    posterior_predictive(container::Union{ABCResults, ADVIResults}, model::Models.AbstractTimescaleModel; show::Bool=true, n_samples::Int=100)

Placeholder function for posterior predictive check plotting.

This function is implemented in the `IntPlottingExt` extension package, which is automatically
loaded when `Plots.jl` is available. The extension provides posterior predictive checking
visualization for both ABC and ADVI inference results.

# Functionality (available when Plots.jl is loaded):
- Creates posterior predictive check plots showing data vs. model predictions
- Works with both `ABCResults` and `ADVIResults` containers
- Shows posterior predictive intervals with uncertainty bands
- Automatically handles ACF and PSD summary statistics with appropriate scaling
- Supports logarithmic scaling when the distance method is logarithmic

# Arguments:
- `container::Union{ABCResults, ADVIResults}`: Results container from inference
- `model::Models.AbstractTimescaleModel`: Model used for inference
- `show::Bool=true`: Whether to display the plot immediately
- `n_samples::Int=100`: Number of posterior samples to use for prediction

# Requirements:
- Requires `Plots.jl` to be loaded for the extension to activate
- Install with: `using Plots` or `import Plots`

See `IntPlottingExt` documentation for complete details.
"""
function posterior_predictive end

end