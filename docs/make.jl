using ChainRules
using Documenter

makedocs(modules=[ChainRules],
         sitename="ChainRules",
         authors="Jarrett Revels and other contributors",
         pages=["Introduction" => "index.md",
                "Getting Started" => "getting_started.md",
                "ChainRules API Documentation" => "api.md"])

deploydocs(repo="github.com/JuliaDiff/ChainRules.jl.git")
