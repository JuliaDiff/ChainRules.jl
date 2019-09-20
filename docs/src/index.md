```@meta
DocTestSetup = :(using ChainRulesCore, ChainRules)
```

# ChainRules

[ChainRules](https://github.com/JuliaDiff/ChainRules.jl) provides a variety of common utilities that can be used by downstream [automatic differentiation (AD)](https://en.wikipedia.org/wiki/Automatic_differentiation) tools to define and execute forward-, reverse-, and mixed-mode primitives.

### Introduction

ChainRules is all about providing a rich set of rules for differentiation.
When a person learns introductory calculus, they learn that the derivative (with respect to `x`) of `a*x` is `a`, and the derivative of `sin(x)` is `cos(x)`, etc.
And they learn how to combine simple rules, via [the chain rule](https://en.wikipedia.org/wiki/Chain_rule), to differentiate complicated functions.
ChainRules is a programmatic repository of that knowledge, with the generalizations to higher dimensions.

[Autodiff (AD)](https://en.wikipedia.org/wiki/Automatic_differentiation) tools roughly work by reducing a problem down to simple parts that they know the rules for, and then combining those rules.
Knowing rules for more complicated functions speeds up the autodiff process as it doesn't have to break things down as much.
For an

**ChainRules is an AD-independent collection of rules to use in a differentiation system.**

### `rrule` and `frule`

!!! terminology "`rrule` and `frule`"
    `rrule` and `frule` are ChainRules specific terms.
    Their exact functioning is fairly ChainRules specific, though other tools have similar functions.
    The core notion is sometimes called _custom AD primitives_, _custom adjoints_, _custom sensitivities_.

The rules are encoded as `rrules` and `frules`, for use in reverse-mode and forward-mode differentiation respectively.

The `rrule` for some function `foo`, which takes the positional argument `args` and keyword argument `kwargs`, is written:

```julia
function rrule(::typeof(foo), args; kwargs...)
    ...
    return y, pullback
end
```
where `y` must be equal to `foo(args; kwargs...)`, and `pullback` is a function to propagate the derivative information backwards at that point (more later).

Similarly, the `frule` is written:
```julia
function frule(::typeof(foo), args; kwargs...)
    ...
    return y, pushforward
end
```
again `y = foo(args; kwargs...)`, and `pushforward` is a function to propagate the derivative information forwards at that point (more later).

Almost always the _pushforward_/_pullback_ will be declared locally within the `ffrule`/`rrule`, and will be a _closure_ over some of the other arguments.

### The propagators: pushforward and pullback

!!! terminology "pushforward and pullback"

    _Pushforward_ and _pullback_ are fancy words that the autodiff community adopted from Differential Geometry.
    The are broadly in agreement with the use of [pullback](https://en.wikipedia.org/wiki/Pullback_(differential_geometry)) and [pushforward](https://en.wikipedia.org/wiki/Pushforward_(differential)) in differential geometry.
    But any geometer will tell you these are the super-boring flat cases. Some will also frown at you.
    Other terms that may be used include for _pullback_ the **backpropagator**, and by analogy for _pushforward_ the **forwardpropagator**, thus these are the _propagators_.
    These are also good names because effectively they propagate wiggles and wobbles through them, via the chainrule.
    (the term **backpropagator** may originate with ["Lambda The Ultimate Backpropagator"](http://www-bcl.cs.may.ie/~barak/papers/toplas-reverse.pdf) by Bearlmutter and Siskind, 2008)

#### Core Idea

 - The **pushforward** takes a wiggle in the _input space_, and tells what wobble you would create in the output space, by passing it through the function.
 - The **pullback** takes a wobble in the _output space_, and tells you what wiggle you would need to make in the _input_ space to achieve it.

#### The anatomy of pushforward and pullback

For our function `foo(args...; kwargs) = Y`:

The pushforward is a function:

```julia
function pushforward(Δself, Δargs...)
    ...
    return ∂Y
end
```

The input to the pushforward is often called the _pertubation_.
If the function is `y = f(x)` often the pushforward will be written `ẏ = pushforward(ḟ, ẋ)`.

!!! note

    There is one `Δarg` per `arg` to the original function.
    The `Δargs` are similar in type/structure to the corresponding inputs `args` (`Δself` is explained below).
    The `∂Y` are similar in type/structure to the original function's output `Y`.
    In particular if that function returned a tuple then `∂Y` will be a tuple of same size.

The pullback is a function:

```julia
function pullback(ΔY)
    ...
    return ∂self, ∂args...
end
```

The input to the pullback is often called the _seed_.
If the function is `y = f(x)` often the pullback will be written `x̄ = pullback(ȳ)`.

!!! note

    The pullback returns one `∂arg` per `arg` to the original function, plus one for the fields of the function itself (explained below).

!!! terminology
    Sometimes _perturbation_, _seed_, and _sensitivity_ will be used interchangeably, depending on task/subfield (sensitivity analysis and perturbation theory are apparently very big on just calling everything _sensitivity_ or _perturbation_ respectively.)
    At the end of the day, they are all _wiggles_ or _wobbles_.

### Self derivative `Δself`, `∂self` etc.

!!! terminology
    To my knowledge there is no standard terminology for this.
    Other good names might be `Δinternal`/`∂internal`.

From the mathematical perspective, one may have been wondering what all this `Δself`, `∂self` is.
After all a function with two inputs, say `f(a, b)`, only has two partial derivatives:
``\dfrac{∂f}{∂a}``, ``\dfrac{∂f}{∂b}``.
Why then does a `pushforward` take in this extra `Δself`, and why does a `pullback` return this extra `∂self`?

The reason is that in Julia the function `f` may itself have internal fields.
For example a closure has the fields it closes over; a callable object (i.e. a functor) like a `Flux.Dense` has the fields of that object.

**Thus every function is treated as having the extra implicit argument `self`, which captures those fields.**
So every `pushforward` takes in an extra argument, which is ignored unless the original function had fields.
In is common to write `function foo_pushforward(_, Δargs...)` in the case when `foo` does not have fields.
Similarly every `pullback` return an extra `∂self`, which for things without fields is the constant `NO_FIELDS`, indicating there are no fields within the function itself.

#### Pushforward / Pullback summary

- **Pushforward:**
    - returned by `ffrule`
    - takes input space wiggles, gives output space wobbles
    - 1 argument per original function argument + 1 for the function itself
    - 1 return per original function return

- **Pullback**
   - returned by `rrule`
   - takes output space wobbles, gives input space wiggles
   - 1 argument per original function return
   - 1 return per original function argument + 1 for the function itself

#### Pushforward/Pullback and Total Derivative/Gradient

The most trivial use of `frule` and returned `pushforward` is to calculate the [Total Derivative](https://en.wikipedia.org/wiki/Total_derivative):

```julia
y, f_pushforward = frule(f, a, b, c)
ẏ = f_pushforward(1, 1, 1, 1)  # for appropriate `1`-like perturbation.
```

Then we have that `ẏ` is the _total derivative_ of `f` at `(a, b, c)`, written mathematically as ``df_{(a,b,c)}``

Similarly, the most trivial use of `rrule` and returned `pullback` is to calculate the [Gradient](https://en.wikipedia.org/wiki/Gradient):

```julia
y, f_pullback = rrule(f, a, b, c)
∇f = f_pullback(1)  # for appropriate `1`-like seed.
s̄, ā, b̄, c̄ = ∇f
```
Then we have that `∇f` is the _gradient_ of `f` at `(a, b, c)`.
And we thus have the partial derivatives ``f̄ = \dfrac{∂f}{∂f}``, ``ā` = \dfrac{∂f}{∂a}``, ``b̄ = \dfrac{∂f}{∂b}``, ``c̄ = \dfrac{∂f}{∂c}``, including the and the self-partial derivative, ``f̄``.

### Differentials

The values that come back from pullbacks or pushforwards are not always the same type as the input/outputs of the original function.
They are differentials; differency-equivalents.
A differential might be such a regular type, like a `Number`, or a `Matrix`, or it might be one of the `AbstractDifferential` subtypes.

Differentials support a number of operations.
Most importantly: `+` and `*` which lets them act as mathematically objects.
And `extern` which converts `AbstractDifferential` types into a conventional non-ChainRules type.

The most important `AbstractDifferential`s when getting started are the ones about avoiding work:

 - `Thunk`: this is a deferred computation. A thunk is a [word for a zero argument closure](https://en.wikipedia.org/wiki/Thunk). A computation wrapped in a `@thunk` doesn't get evaluated until `extern` is called on the `Thunk`. More on thunks later.
 - `One`, `Zero`: There are special representations of `1` and `0`. They do great things around avoiding expanding `Thunks` in multiplication and (for `Zero`) addition.

#### Other `AbstractDifferential`s: don't worry about them right now
 - `Wirtinger`: it is complex. The docs need to be better. [Read the links in this issue](https://github.com/JuliaDiff/ChainRulesCore.jl/issues/40).
 - `Casted`: it implements broadcasting mechanics. See [#10](https://github.com/JuliaDiff/ChainRulesCore.jl/issues/10)
 - `InplaceableThunk`: it is like a Thunk but it can do `store!` and `accumulate!` in-place.

 -------------------------------

## Example of using ChainRules directly.

While ChainRules is largely intended as a backend for autodiff systems, it can be used directly.
In fact, this can be very useful if you can constrain the code you need to differentiate to only use things that have rules defined for.
This was once how all neural network code worked.

Using ChainRules directly also helps get a feel for it.

```julia
using ChainRules

function foo(x)
    a = sin(x)
    b = 2a
    c = asin(b)
    return c
end

#### Find dfoo/dx via rrules

# First the forward pass, accumulating rules
x = 3;
a, a_pullback = rrule(sin, x);
b, b_pullback = rrule(*, 2, a);
c, c_pullback = rrule(asin, b)

# Then the backward pass calculating gradients
c̄ = 1;  # ∂c/∂c
_, b̄ = c_pullback(extern(c̄));     # ∂c/∂b
_, _, ā = b_pullback(extern(b̄));  # ∂c/∂a
_, x̄ = a_pullback(extern(ā));     # ∂c/∂x = ∂f/∂x
extern(x̄)
# -2.0638950738662625

#### Find dfoo/dx via frules

# Unlike with rrule, we can interleave evaluation and derivative evaluation
x = 3;
ẋ = 1;  # ∂x/∂x
nofields = NamedTuple();  # ∂self/∂self

a, a_pushforward = frule(sin, x);
ȧ = a_pushforward(nofields, extern(ẋ));     # ∂a/∂x

b, b_pushforward = frule(*, 2, a);
ḃ = b_pushforward(nofields, 0, extern(ȧ));  # ∂b/∂x = ∂b/∂a⋅∂a/∂x

c, c_pushforward = frule(asin, b);
ċ = c_pushforward(nofields, extern(ḃ));     # ∂c/∂x = ∂c/∂b⋅∂b/∂x = ∂f/∂x
extern(ċ)
# -2.0638950738662625

#### Find dfoo/dx via finite-differences

using FiniteDifferences
central_fdm(5, 1)(foo, x)
# -2.0638950738670734

#### Find dfoo/dx via finite-differences ForwardDiff.jl
using ForwardDiff
ForwardDiff.derivative(foo, x)
# -2.0638950738662625

#### Find dfoo/dx via finite-differences Zygote.jl
using Zygote
Zygote.gradient(foo, x)
# (-2.0638950738662625,)
```

 -------------------------------

## On writing good `rrule` / `frule` methods

### Return Zero or One

Rather than `0` or `1` or even rather than `zeros(n)`, `ones(n)`, or the identity matrix `I`.

### Use `Thunk`s appropriately:

If work is only required for one of the returned differentials, then it should be wrapped in a `@thunk` (potentially using a `begin`-`end` block).

If there are multiple return values, their computation should almost always be wrapped in a `@thunk`.

Do _not_ wrap _variables_ in a `@thunk`, wrap the _computations_ that fill those variables in `@thunk`:

```julia
# good:
∂A = @thunk(foo(x))
return ∂A

# bad:
∂A = foo(x)
return @thunk(∂A)
```
In the bad example `foo(x)` gets computed eagerly, and all that the thunk is doing is wrapping the already calculated result in a function that returns it.

### Be careful with using `adjoint` when you mean `transpose`

Remember for complex numbers `a'` (i.e. `adjoint(a)`) takes the complex conjugate.
Instead you probably want `transpose(a)`.

While there are arguments that for reverse-mode taking the adjoint is correct, it is not currently the behavior of ChainRules to do so.
Feel free to open an issue to discuss it.

### Code Style

Use named local functions for the `pushforward`/`pullback`:

```julia
# good:
function frule(::typeof(foo), x)
    Y = foo(x)
    function foo_pushforward(_, ẋ)
        return bar(ẋ)
    end
    return Y, foo_pushforward
end

# bad:
function frule(::typeof(foo), x)
    return foo(x), (_, ẋ) -> bar(ẋ)
end
```

While this is more verbose, it ensures that if an error is thrown during the `pullback`/`pushforward` the [`gensym`](https://docs.julialang.org/en/v1/base/base/#Base.gensym) name of the local function will include the name you gave it.
This makes it a lot simpler to debug from the stacktrace.

### Write tests

There are fairly decent tools for writing tests based on [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl).
They are in [`tests/test_utils.jl`](https://github.com/JuliaDiff/ChainRules.jl/blob/master/test/test_util.jl).
Take a look at existing test and you should see how to do stuff.

!!! warning
    Use finite differencing to test derivatives.
    Don't use analytical derivations for derivatives in the tests!
    Since the rules are analytic expressions, re-writing those same expressions in the tests
    can not be an effective way to test, and will give misleading test coverage.

### CAS systems are your friends.

It is very easy to check gradients or derivatives with a computer algebra system (CAS) like [WolframAlpha](https://www.wolframalpha.com/input/?i=gradient+atan2%28x%2Cy%29).

------------------------------------------

## FAQ

### What is up with the different symbols?

 - `Δx` is the input to a propagator, (i.e a _seed_ for a _pullback_; or a _perturbation_ for a _pushforward_)
 - `∂x` is the output of a propagator
 - `dx` could be anything, including a pullback. It really should not show up outside of tests.
 - `v̇` is a derivative of the input moving forward: ``v̇ = \frac{∂v}{∂x}`` for input ``x``, intermediate value ``v``.
 - `v̄` is a derivative of the output moving backward: ``v̄ = \frac{∂y}{∂v}`` for output ``y``, intermediate value ``v``.
 - `Ω` is often used as the return value of the function having the rule found for. Especially, (but not exclusively.) for scalar functions.
     - `ΔΩ` is thus a seed for the pullback.
     - `∂Ω` is thus the output of a pushforward.

### Why does `frule` and `rrule` return the function evaluation?

You might wonder why `frule(f, x)` returns `f(x)` and the pushforward for `f` at `x`, and similarly for `rrule` returing `f(x)` and the pullback for `f` at `x`.
Why not just return the pushforward/pullback, and let the user call `f(x)` to get the answer seperately?

Their are two reasons the rules also create the `f(x)`.
1. For some rules the output value is used in the definition of its propagator. For example `tan`.
2. For some rules an alternative way of calculating `f(x)` can give the same answer while also generating intermediate values that can be used in the calculations within the propagator.
