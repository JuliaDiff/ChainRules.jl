using ChainRules
using Documenter

makedocs(modules=[ChainRules],
         sitename="ChainRules.jl",
         authors="Jarrett Revels and other contributors",
         pages=["Home" => "index.md"])

deploydocs(repo="github.com/JuliaDiff/ChainRules.jl.git")
