using ChainRules
using ChainRulesCore
using Documenter

@show ENV

makedocs(
    modules=[ChainRules, ChainRulesCore],
    format=Documenter.HTML(prettyurls=false, assets = ["assets/chainrules.css"]),
    sitename="ChainRules",
    authors="Jarrett Revels and other contributors",
    pages=[
        "Introduction" => "index.md",
        "Getting Started" => "getting_started.md",
        "API" => "api.md",
    ],
)

const repo="github.com/JuliaDiff/ChainRules.jl.git"
const PR = get(ENV, "TRAVIS_PULL_REQUEST", "false")
if PR == "false"
    # Normal case, only deply docs if merging to master or release tagged
    deploydocs(repo=repo)
else
    @info "Deploying review docs for PR #$PR"
    # TODO: remove most of this once https://github.com/JuliaDocs/Documenter.jl/issues/1131 is resolved

    # Overwrite Documenter's function for generating the versions.js file
    foreach(Base.delete_method, methods(Documenter.Writers.HTMLWriter.generate_version_file))
    Documenter.Writers.HTMLWriter.generate_version_file(_, _) = nothing
    # Overwrite necessary environment variables to trick Documenter to deploy
    ENV["TRAVIS_PULL_REQUEST"] = "false"
    ENV["TRAVIS_BRANCH"] = "master"

    deploydocs(
        devurl="preview-PR$(PR)",
        repo=repo,
    )
end
