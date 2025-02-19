push!(LOAD_PATH, "../src/")
using Documenter
using IntrinsicTimescales

makedocs(sitename="IntrinsicTimescales.jl",
         format=Documenter.HTML(),
         pages=["Getting Started" => "home.md",
         "Practice" => Any["practice/practice_intro.md",
         "practice/practice_1_acf.md",
         "practice/practice_2_acw.md",
         "practice/practice_3_ou.md",
         "practice/practice_4_psd.md"],
         "Theory" => "theory/theory.md",
         "Implementation" => Any[
            "Model-Free Timescale Estimation" => "acw.md",
             "Simulation Based Timescale Estimation" => ["Overview" => "simbasedinference.md",
                 "One Timescale Model" => "one_timescale.md",
                 "One Timescale Model with Missing Data" => "one_timescale_with_missing.md",
                 "One Timescale Model with Oscillations" => "one_timescale_and_osc.md",
                 "One Timescale Model with Oscillations and Missing Data" => "one_timescale_and_osc_with_missing.md",
                 "Model Fitting and Parameters" => "fit_parameters.md",
                 "Results" => "fit_result.md"
             ]],
             "API" => "index.md"])


# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/duodenum96/IntrinsicTimescales.jl.git",
)
