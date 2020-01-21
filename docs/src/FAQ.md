# FAQ

## What is up with the different symbols?

### `Δx`, `∂x`, `dx`
ChainRules uses these perhaps atyptically.
As a notation that is the same across propagators, regardless of direction (incontrast see `ẋ` and `x̄` below).

 - `Δx` is the input to a propagator, (i.e a _seed_ for a _pullback_; or a _perturbation_ for a _pushforward_)
 - `∂x` is the output of a propagator
 - `dx` could be either


### dots and bars: ``\dot{y} = \dfrac{∂y}{∂x} = \overline{x}``
 - `v̇` is a derivative of the input moving forward: ``v̇ = \frac{∂v}{∂x}`` for input ``x``, intermediate value ``v``.
 - `v̄` is a derivative of the output moving backward: ``v̄ = \frac{∂y}{∂v}`` for output ``y``, intermediate value ``v``.

### others
 - `Ω` is often used as the return value of the function. Especially, but not exclusively, for scalar functions.
     - `ΔΩ` is thus a seed for the pullback.
     - `∂Ω` is thus the output of a pushforward.


## Why does `rrule` return the primal function evaluation?
You might wonder why `frule(f, x)` returns `f(x)` and the derivative of `f` at `x`, and similarly for `rrule` returning `f(x)` and the pullback for `f` at `x`.
Why not just return the pushforward/pullback, and let the user call `f(x)` to get the answer separately?

There are three reasons the rules also calculate the `f(x)`.
1. For some rules an alternative way of calculating `f(x)` can give the same answer while also generating intermediate values that can be used in the calculations required to propagate the derivative.
2. For many `rrule`s the output value is used in the definition of the pullback. For example `tan`, `sigmoid` etc.
3. For some `frule`s there exists a single, non-separable operation that will compute both derivative and primal result. For example many of the methods for [differential equation sensitivity analysis](https://docs.juliadiffeq.org/latest/analysis/sensitivity/#sensitivity-1).

## Where are the derivatives for keyword arguments?
_pullbacks_ do not return a sensitivity for keyword arguments;
similarly _pushfowards_ do not accept a perturbation for keyword arguments.
This is because in practice functions are very rarely differentiable with respect to keyword arguments.
As a rule keyword arguments tend to control side-effects, like logging verbosity,
or to be functionality changing to perform a different operation, e.g. `dims=3`, and thus not differentiable.
To the best of our knowledge no Julia AD system, with support for the definition of custom primitives, supports differentiating with respect to keyword arguments.
At some point in the future ChainRules may support these. Maybe.


## What is the difference between `Zero` and `DoesNotExist` ?
`Zero` and `DoesNotExist` act almost exactly the same in practice: they result in no change whenever added to anything.
Odds are if you write a rule that returns the wrong one everything will just work fine.
We provide both to allow for clearer writing of rules, and easier debugging.

`Zero()` represents the fact that if one perturbs (adds a small change to) the matching primal there will be no change in the behavour of the primal function.
For example in `fst(x,y) = x`, then the derivative of `fst` with respect to `y` is `Zero()`.
`fst(10, 5) == 10` and if we add `0,1` to `5` we still get `fst(10, 5.1)=10`.

`DoesNotExist()` represents the fact that if one perturbs the matching primal, the primal function will now error.
For example in `access(xs, n) = xs[n]` then the derivative of `access` with respect to `n` is `DoesNotExist()`.
`access([10, 20, 30], 2) = 20`, but if we add `0.1` to `2` we get `access([10, 20, 30], 2.1)` which errors as indexing can't be applied at fractional indexes.


## When to use ChainRules vs ChainRulesCore?

[ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl) is a light-weight dependency for defining rules for functions in your packages, without you needing to depend on ChainRules itself. It has no dependencies of its own.

[ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) provides the full functionality, in particular it has all the  rules for Base Julia and the standard libraries. Its thus a much heavier package to load.

If you only want to define rules, not use them then you probably only want to load ChainRulesCore.
AD systems making use of ChainRules should load ChainRules (rather than ChainRulesCore).

## Where should I put my rules?
In general, we recommend adding custom sensitivities to your own packages with ChainRulesCore, rather than adding them to ChainRules.jl.

A few packages currently SpecialFunctions.jl and NaNMath.jl are in ChainRules.jl but this is a short-term measure.
