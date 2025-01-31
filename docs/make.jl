push!(LOAD_PATH,"../src/")
using Documenter, INT


makedocs(sitename="INT", pages = ["Home" => "intro_page.md", 
                                    "Practice" => ["practice/practice_intro.md", 
                                    "practice/practice_1_acf.md"],
                                    "API" => "index.md"])

