# Getting Started

[ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl) is a light-weight dependency for defining sensitivities for functions in your packages, without you needing to depend on ChainRules itself. It has no dependencies of its own.

[ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) provides the full functionality, including sensitivities for Base Julia and standard libraries.
Sensitivities for some other packages, currently SpecialFunctions.jl and NaNMath.jl, will also be loaded if those packages are in your environment.
In general, we recommend adding custom sensitivities to your own packages with ChainRulesCore, rather than adding them to ChainRules.jl.

## Defining Custom Sensitivities

TODO

## Forward-Mode vs. Reverse-Mode Chain Rule Evaluation

TODO

## Real Scalar Differentiation Rules

TODO

## Complex Scalar Differentiation Rules

TODO

## Non-Scalar Differentiation Rules

TODO
