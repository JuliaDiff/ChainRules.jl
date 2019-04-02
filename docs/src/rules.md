# Differentiation Rules

```@docs
ChainRules.AbstractRule
```

## Forward and Reverse Rules

Two workhorse functions in this package are `frule`, short for "forward rule," and `rrule`,
short for "reverse rule."
These correspond to forward- and reverse-mode update rules, respectively.

```@docs
ChainRules.frule
ChainRules.rrule
```

In a sense, the fallback definition for, for example, `frule`, should just be "compute
the derivative using forward-mode AD."
This is necessary to enable mixed-mode rules where e.g. `frule` is used within an `rrule`
definition.
One example of this is `broadcast`ed functions, which may themselves not be forward-mode
_primitives_, but are forward-mode _differentiable_.

By design, ChainRules is decoupled from any particular AD implementation.
That begs the question of how to fall back when there isn't a primitive defined.

Some AD packages may choose to extend `frule`/`rrule` to use their own implementation.
However, this will not play well with other packages which may do the same thing, thereby
causing issues with package load order dependency for downstream users.

As it turns out, [Cassette](https://github.com/jrevels/Cassette.jl) solves this problem
nicely by allowing AD package authors to extend the fallback definitions with respect to
their own Cassette contexts.
Here is an example of that using the [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl)
package:

```julia
using ChainRules, ForwardDiff, Cassette

Cassette.@context MyChainRuleCtx

# ForwardDiff itself can call `myfrule` instead of `frule` to utilize ForwardDiff-injected
# ChainRules infrastructure
myfrule(args...) = Cassette.overdub(MyChainRuleCtx(), frule, args...)

function Cassette.execute(::MyChainRuleCtx, ::typeof(frule), f, x::Number)
    r = frule(f, x)
    if r === nothing
        fx = f(x)
        dx = Rule(Δx -> ForwardDiff.derivative(f, x) * Δx)
    else
        fx, dx = r
    end
    return fx, dx
end
```

## Defining Rules for Scalars

Defining differentiation rules for scalar-valued functions can be accomplished easily using
the `@scalar_rule` macro.

```@docs
ChainRules.@scalar_rule
```

A very simple example of this is for the `sin` and `cos` functions:

```julia
@scalar_rule(sin(x), cos(x))
@scalar_rule(cos(x), -sin(x))
```
