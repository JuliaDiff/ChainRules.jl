## Package Glue Code

In the ideal world, everyone would write ChainRules for their functions
in the packages where they are defined.
By depending only on [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl)
We do not live in an ideal world, so some of those definitions live here.
In the long-term the plan is to move them out of this repo.
