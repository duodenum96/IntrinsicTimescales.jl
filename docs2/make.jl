push!(LOAD_PATH, "../src/")
using Documenter
using INT

makedocs(sitename="INT.jl",
         #  format=Documenter.HTML(),
         #  # modules = [INT],
         #  checkdocs=:exports,
         pages=["Model-Free Timescale Estimation" => "acw.md",
             "Simulation Based Timescale Estimation" => ["simbasedinference.md",
                 "One Timescale Model" => "one_timescale.md"],
             "API" => "index.md"])

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
