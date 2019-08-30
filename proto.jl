using Pkg: @pkg_str
pkg"activate /Users/oxinabox/JuliaEnvs/ChainRulesWorld/"
using Revise


include("/Users/oxinabox/JuliaEnvs/ChainRulesWorld/ChainRulesCore.jl/test/runtests.jl")

include("/Users/oxinabox/JuliaEnvs/ChainRulesWorld/ChainRules.jl/test/runtests.jl")


using FiniteDifferences
using Test
using ChainRules
using Random

const accumulate = ChainRules.ChainRulesCore.accumulate
const accumulate! = ChainRules.ChainRulesCore.accumulate!
const add = ChainRules.ChainRulesCore.add
includet("/Users/oxinabox/JuliaEnvs/ChainRulesWorld/ChainRules.jl/test/test_util.jl")

#==
Test Summary:               | Pass  Fail  Error  Total
ChainRules                  | 2271    38     88   2397

==#
using MacroTools
using MacroTools: textwalk

code = """
Rule(x -> -sin(x))
Rule(x -> 1 + tan(x)^2)
""";

after = textwalk(code) do expr
    @capture(expr, Rule(v_)) && return MacroTools.postwalk(MacroTools.unblock, v)
    return expr
end

println(after)

##############


code = """
y->-sin(x)
""";

after = textwalk(code) do expr
    @capture(expr, v_) && return v
end

println(after)
