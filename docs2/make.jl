push!(LOAD_PATH,"../src/")
using Documenter
using INT

makedocs(
    sitename = "INT",
    format = Documenter.HTML(),
    # modules = [INT],
    checkdocs = :exports
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
