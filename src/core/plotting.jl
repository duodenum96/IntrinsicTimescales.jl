module Plotting

using Plots
colorpalette = palette(:Catppuccin_mocha)[[4, 5, 7, 9, 3, 10, 13]]

function plot(container::ACWContainer; only_acf::Bool=false, only_psd::Bool=false, show::Bool=true)
    # Check if we have data to plot
    if only_acf && isnothing(container.acf)
        error("ACF data not available in container")
    elseif only_psd && isnothing(container.psd)
        error("PSD data not available in container")
    end

    if !isnothing(container.acf)
        if ndims(container.acf) > 2
            throw("We don't support plotting for more than 2 dimensions. Use a for-loop.")
        end
    end

    # Determine number of subplots needed
    n_plots = ((!only_psd && !isnothing(container.acf)) ? 1 : 0) + 
              ((!only_acf && !isnothing(container.psd)) ? 1 : 0)
    
    if n_plots == 0
        error("No data available to plot")
    end

    # Create figure layout
    p = n_plots == 2 ? plot(layout=(1,2)) : plot()
    current_plot = 1

    # Plot ACF if available and requested
    if !only_psd && !isnothing(container.acf)
        plot!(p[current_plot], container.lags, container.acf, 
              label="ACF", color=colorpalette[1],
              xlabel="Lag (s)", ylabel="Autocorrelation",
              title="Autocorrelation Function")
        current_plot += 1
    end

    # Plot PSD if available and requested
    if !only_acf && !isnothing(container.psd)
        plot!(p[current_plot], container.freqs, container.psd, 
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



end

