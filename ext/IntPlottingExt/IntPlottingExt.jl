module IntPlottingExt

using IntrinsicTimescales
using Plots
using Statistics

export acwplot, posterior_predictive

colorpalette = palette(:Catppuccin_mocha)[[4, 5, 7, 9, 3, 10, 13]]

function IntrinsicTimescales.acwplot(acwresults::ACWResults; only_acf::Bool=false, only_psd::Bool=false, show::Bool=true)
    # Check if we have data to plot
    if only_acf && isnothing(acwresults.acf)
        error("ACF data not available in container")
    elseif only_psd && isnothing(acwresults.psd)
        error("PSD data not available in container")
    end

    if !isnothing(acwresults.acf)
        if ndims(acwresults.acf) > 2
            throw("We don't support plotting for more than 2 dimensions. Use a for-loop.")
        end
    end

    # Determine number of subplots needed
    n_plots = ((!only_psd && !isnothing(acwresults.acf)) ? 1 : 0) + 
              ((!only_acf && !isnothing(acwresults.psd)) ? 1 : 0)
    
    if n_plots == 0
        error("No data available to plot")
    end

    # Create figure layout
    p = n_plots == 2 ? plot(layout=(1,2)) : plot()
    current_plot = 1

    # Plot ACF if available and requested
    if !only_psd && !isnothing(acwresults.acf)
        plot!(p[current_plot], acwresults.lags, acwresults.acf', 
              palette=colorpalette,
              xlabel="Lag (s)", ylabel="Autocorrelation",
              title="Autocorrelation Function", label="")
        plot!(p[current_plot], acwresults.lags, mean(acwresults.acf, dims=1)', 
              color=:black, linewidth=2, label="")
        current_plot += 1
    end

    # Plot PSD if available and requested
    if !only_acf && !isnothing(acwresults.psd)
        plot!(p[current_plot], acwresults.freqs, acwresults.psd', 
              label="PSD", color=colorpalette[2],
              xlabel="Frequency (Hz)", ylabel="Power",
              title="Power Spectral Density",
              xscale=:log10, yscale=:log10)
    end
    if show
        display(p)
    end
    return p
end

"""
    posterior_predictive(container::ABCResults, model::Models.AbstractTimescaleModel; show::Bool=true)

Plot posterior predictive check for ABC results. Shows the data summary statistics (ACF or PSD)
with posterior predictive samples overlaid.

# Arguments
- `container::ABCResults`: Container with ABC results
- `model::Models.AbstractTimescaleModel`: Model used for inference
- `show::Bool=true`: Whether to display the plot
- `n_samples::Int=100`: Number of posterior samples to use for prediction

# Returns
- Plot object
"""
function IntrinsicTimescales.posterior_predictive(container::ABCResults, model::Models.AbstractTimescaleModel; 
             show::Bool=true, n_samples::Int=100)
    
    # Randomly sample from posterior
    n_params = size(container.final_theta, 2)
    sample_indices = rand(1:size(container.final_theta, 1), n_samples)
    theta_samples = container.final_theta[sample_indices, :]
    
    # Generate predictions for each sample
    predictions = zeros(n_samples, length(model.data_sum_stats))
    for i in 1:n_samples
        sim_data = Models.generate_data(model, theta_samples[i, :])
        predictions[i, :] = Models.summary_stats(model, sim_data)
    end
    
    # Calculate mean and quantiles of predictions
    pred_mean = mean(predictions, dims=1)[:]
    pred_lower = [quantile(predictions[:, i], 0.025) for i in 1:size(predictions, 2)]
    pred_upper = [quantile(predictions[:, i], 0.975) for i in 1:size(predictions, 2)]
    
    # Create plot
    p = plot()
    
    # Plot posterior predictive interval
    if model.summary_method == :acf
        if model.distance_method == :logarithmic
            plot!(p, model.lags_freqs, pred_mean,
                  ribbon=(pred_mean - pred_lower, pred_upper - pred_mean),
                  fillalpha=0.3, color=colorpalette[1], label="Posterior predictive",
                  yscale=:log10)
            plot!(p, model.lags_freqs, model.data_sum_stats,
                  color=:black, linewidth=2, label="Data",
                  xlabel="Lag (s)", ylabel="Autocorrelation",
                  title="Posterior Predictive Check - ACF",
                  yscale=:log10)
        else
            plot!(p, model.lags_freqs, pred_mean,
                  ribbon=(pred_mean - pred_lower, pred_upper - pred_mean),
                  fillalpha=0.3, color=colorpalette[1], label="Posterior predictive")
            plot!(p, model.lags_freqs, model.data_sum_stats,
                  color=:black, linewidth=2, label="Data",
                  xlabel="Lag (s)", ylabel="Autocorrelation",
                  title="Posterior Predictive Check - ACF")
        end
              
    elseif model.summary_method == :psd
        # For PSD, use log scale if distance method is logarithmic
        if model.distance_method == :logarithmic
            plot!(p, model.lags_freqs, pred_mean,
                  ribbon=(pred_mean - pred_lower, pred_upper - pred_mean),
                  fillalpha=0.3, color=colorpalette[1], label="Posterior predictive",
                  xscale=:log10, yscale=:log10)
            plot!(p, model.lags_freqs, model.data_sum_stats,
                  color=:black, linewidth=2, label="Data",
                  xlabel="Frequency (Hz)", ylabel="Power",
                  title="Posterior Predictive Check - PSD")
        else
            plot!(p, model.lags_freqs, pred_mean,
                  ribbon=(pred_mean - pred_lower, pred_upper - pred_mean),
                  fillalpha=0.3, color=colorpalette[1], label="Posterior predictive")
            plot!(p, model.lags_freqs, model.data_sum_stats,
                  color=:black, linewidth=2, label="Data",
                  xlabel="Frequency (Hz)", ylabel="Power",
                  title="Posterior Predictive Check - PSD")
        end
    end
    
    if show
        display(p)
    end
    return p
end

"""
    posterior_predictive(container::ADVIResults, model::Models.AbstractTimescaleModel; show::Bool=true)


Plot posterior predictive check for ADVI results. Shows the data summary statistics (ACF or PSD)
with posterior predictive samples overlaid.

# Arguments
- `container::ADVIResults`: Container with ADVI results
- `model::Models.AbstractTimescaleModel`: Model used for inference
- `show::Bool=true`: Whether to display the plot
- `n_samples::Int=100`: Number of posterior samples to use for prediction

# Returns
- Plot object
"""
function IntrinsicTimescales.posterior_predictive(container::ADVIResults, model::Models.AbstractTimescaleModel; 
             show::Bool=true, n_samples::Int=100)
    

    # Randomly sample from posterior
    sample_indices = rand(1:size(container.samples, 2), n_samples)
    theta_samples = container.samples[:, sample_indices]'  # Transpose to match ABC format
    
    # Generate predictions for each sample
    predictions = zeros(n_samples, length(model.data_sum_stats))
    for i in 1:n_samples
        sim_data = Models.generate_data(model, theta_samples[i, :])
        predictions[i, :] = Models.summary_stats(model, sim_data)
    end
    
    # Calculate mean and quantiles of predictions
    pred_mean = mean(predictions, dims=1)[:]
    pred_lower = [quantile(predictions[:, i], 0.025) for i in 1:size(predictions, 2)]
    pred_upper = [quantile(predictions[:, i], 0.975) for i in 1:size(predictions, 2)]
    
    # Create plot
    p = plot()
    
    # Plot posterior predictive interval
    if model.summary_method == :acf
        if model.distance_method == :logarithmic
            plot!(p, model.lags_freqs, pred_mean,
                  ribbon=(pred_mean - pred_lower, pred_upper - pred_mean),
                  fillalpha=0.3, color=colorpalette[1], label="Posterior predictive",
                  yscale=:log10)
            plot!(p, model.lags_freqs, model.data_sum_stats,
                  color=:black, linewidth=2, label="Data",
                  xlabel="Lag (s)", ylabel="Autocorrelation",
                  title="Posterior Predictive Check - ACF",
                  yscale=:log10)
        else
            plot!(p, model.lags_freqs, pred_mean,
                  ribbon=(pred_mean - pred_lower, pred_upper - pred_mean),
                  fillalpha=0.3, color=colorpalette[1], label="Posterior predictive")
            plot!(p, model.lags_freqs, model.data_sum_stats,
                  color=:black, linewidth=2, label="Data",
                  xlabel="Lag (s)", ylabel="Autocorrelation",
                  title="Posterior Predictive Check - ACF")
        end
              
    elseif model.summary_method == :psd
        # For PSD, use log scale if distance method is logarithmic
        if model.distance_method == :logarithmic
            plot!(p, model.lags_freqs, pred_mean,
                  ribbon=(pred_mean - pred_lower, pred_upper - pred_mean),
                  fillalpha=0.3, color=colorpalette[1], label="Posterior predictive",
                  xscale=:log10, yscale=:log10)
            plot!(p, model.lags_freqs, model.data_sum_stats,
                  color=:black, linewidth=2, label="Data",
                  xlabel="Frequency (Hz)", ylabel="Power",
                  title="Posterior Predictive Check - PSD")
        else
            plot!(p, model.lags_freqs, pred_mean,
                  ribbon=(pred_mean - pred_lower, pred_upper - pred_mean),
                  fillalpha=0.3, color=colorpalette[1], label="Posterior predictive")
            plot!(p, model.lags_freqs, model.data_sum_stats,
                  color=:black, linewidth=2, label="Data",
                  xlabel="Frequency (Hz)", ylabel="Power",
                  title="Posterior Predictive Check - PSD")
        end
    end
    
    if show
        display(p)
    end
    return p
end

end

