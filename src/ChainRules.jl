module ChainRules

using Base.Broadcast: materialize, materialize!, broadcasted

import NaNMath, SpecialFunctions

include("chain.jl")
include("rules.jl")
include("rules/base.jl")
include("rules/nanmath.jl")
include("rules/specialfunctions.jl")

end # module
