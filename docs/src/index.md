```@meta
DocTestSetup = :(using ChainRulesCore, ChainRules)
```

# ChainRules

[ChainRules](https://github.com/JuliaDiff/ChainRules.jl) provides a variety of common utilities that can be used by downstream [automatic differentiation (AD)](https://en.wikipedia.org/wiki/Automatic_differentiation) tools to define and execute forward-, reverse-, and mixed-mode primitives.

## Introduction

ChainRules is all about providing a rich set of rules for differentiation.
When a person learns introductory calculus, they learn that the derivative (with respect to `x`) of `a*x` is `a`, and the derivative of `sin(x)` is `cos(x)`, etc.
And they learn how to combine simple rules, via [the chain rule](https://en.wikipedia.org/wiki/Chain_rule), to differentiate complicated functions.
ChainRules is a programmatic repository of that knowledge, with the generalizations to higher dimensions.

[Autodiff (AD)](https://en.wikipedia.org/wiki/Automatic_differentiation) tools roughly work by reducing a problem down to simple parts that they know the rules for, and then combining those rules.
Knowing rules for more complicated functions speeds up the autodiff process as it doesn't have to break things down as much.

**ChainRules is an AD-independent collection of rules to use in a differentiation system.**


!!! note "The whole field is a mess for terminology"
    It isn't just ChainRules, it is everyone.
    Internally ChainRules tries to be consistent.
    Help with that is always welcomed.

!!! terminology "Primal"
Often we will talk about something as _primal_.
That means it is related to the original problem, not its derivative.
For example for `y = foo(x)`
`foo` is the _primal_ function,
computing `foo(x)` is doing the _primal_ computation.
`y` is the _primal_ return, and `x` is a _primal_ argument.
`typeof(y)` and `typeof(x)` are both _primal_ types.


## `frule` and `rrule`

!!! terminology "`frule` and `rrule`"
    `frule` and `rrule` are ChainRules specific terms.
    Their exact functioning is fairly ChainRules specific, though other tools have similar functions.
    The core notion is sometimes called _custom AD primitives_, _custom adjoints_, _custom_gradients_, _custom sensitivities_.

The rules are encoded as `frule`s and `rrule`s, for use in forward-mode and reverse-mode differentiation respectively.

The `rrule` for some function `foo`, which takes the positional arguments `args` and keyword arguments `kwargs`, is written:

```julia
function rrule(::typeof(foo), args...; kwargs...)
    ...
    return y, pullback
end
```
where `y` (the primal result) must be equal to `foo(args...; kwargs...)`.
`pullback` is a function to propagate the derivative information backwards at that point.
That pullback function is used like:
`∂self, ∂args... = pullback(Δy)`


Almost always the _pullback_ will be declared locally within the `rrule`, and will be a _closure_ over some of the other arguments, and potentially over the primal result too.

The `frule` is written:
```julia
function frule(::typeof(foo), args..., Δself, Δargs...; kwargs...)
    ...
    return y, ∂Y
end
```
where again `y = foo(args; kwargs...)`,
and `∂Y` is the result of propagating the derivative information forwards at that point.
This propagation is call the pushforward.
One could think of writing `∂Y = pushforward(Δself, Δargs)`, and often we will think of the `frule` as having the primal computation `y = foo(args...; kwargs...)`, and the push-forward `∂Y = pushforward(Δself, Δargs...)`


!!! note "Why `rrule` returns a pullback but `frule` doesn't return a pushforward"
    While `rrule` takes only the arguments to the original function (the primal arguments) and returns a function (the pullback) that operates with the derivative information, the `frule` does it all at once.
    This is because the `frule` fuses the primal computation and the pushforward.
    This is an optimization that allows `frule`s to contain single large operations that perform both the primal computation and the pushforward at the same time (for example solving an ODE).
This operation is only possible in forward mode (where `frule` is used) because the derivative information needed by the pushforward available with the `frule` is invoked -- it is about the primal function's inputs.
    In contrast, in reverse mode the derivative information needed by the pullback is about the primal function's output.
    Thus the reverse mode returns the pullback function which the caller (usually an AD system) keeps hold of until derivative information about the output is available.


## The propagators: pushforward and pullback


!!! terminology "pushforward and pullback"

    _Pushforward_ and _pullback_ are fancy words that the autodiff community recently adopted from Differential Geometry.
    The are broadly in agreement with the use of [pullback](https://en.wikipedia.org/wiki/Pullback_(differential_geometry)) and [pushforward](https://en.wikipedia.org/wiki/Pushforward_(differential)) in differential geometry.
    But any geometer will tell you these are the super-boring flat cases. Some will also frown at you.
    They are also sometimes described in terms of the jacobian:
    The _pushforward_ is _jacobian vector product_ (`jvp`), and _pullback_ is _jacobian transpose vector product_ (`j'vp`).
    Other terms that may be used include for _pullback_ the **backpropagator**, and by analogy for _pushforward_ the **forwardpropagator**, thus these are the _propagators_.
    These are also good names because effectively they propagate wiggles and wobbles through them, via the chain rule.
    (the term **backpropagator** may originate with ["Lambda The Ultimate Backpropagator"](http://www-bcl.cs.may.ie/~barak/papers/toplas-reverse.pdf) by Pearlmutter and Siskind, 2008)

### Core Idea

#### Less formally

 - The **pushforward** takes a wiggle in the _input space_, and tells what wobble you would create in the output space, by passing it through the function.
 - The **pullback** takes wobbliness information with respect to the function's output, and tells the equivalent wobbliness with respect to the functions input.

#### More formally
The **pushforward** of ``f`` takes the _sensitivity_ of the input of ``f`` to a quantity, and gives the _sensitivity_ of the output of ``f`` to that quantity
The **pullback** of ``f`` takes the _sensitivity_ of a quantity to the output of ``f``, and gives the _sensitivity_ of that quantity to the input of ``f``.

### Math
This is all a bit simplified by talking in 1D.

#### Lighter Math
For a chain of expressions:
```
a = f(x)
b = g(a)
c = h(b)
```

The pullback of `g`, which incorporates the knowledge of `∂b/∂a`,
applies the chain rule to go from `∂c/∂b` to `∂c/∂a`.

The pushforward of `g`,  which also incorporates the knowledge of `∂b/∂a`,
applies the chain rule to go from `∂a/∂x` to `∂b/∂x`.

### Heavier Math
If I have some functions: ``g(a)``, ``h(b)`` and ``f(x)=g(h(x))``, and I know
the pullback of ``g``, at ``h(x)`` written: ``\mathrm{pullback}_{g(a)|a=h(x)}``,
and I know the derivative of ``h`` with respect to its input ``b`` at ``g(x)``,
written: ``\left.\dfrac{∂h}{∂b}\right|_{b=g(x)}`` Then I can use the pullback to
find: ``\dfrac{∂f}{∂x}``:

``\dfrac{∂f}{∂x}=\mathrm{\mathrm{pullback}_{g(a)|a=h(x)}}\left(\left.\dfrac{∂h}{∂b}\right|_{b=g(x)}\right).``

If I know the derivative of ``g`` with respect to its input a at ``x``, written:
``\left.\dfrac{∂g}{∂a}\right|_{a=x}``, and I know the pushforward of ``h`` at
``g(x)`` written: ``\mathrm{pushforward}_{h(b)|b=g(x)}``. Then I can use the
pushforward to find ``\dfrac{∂f}{∂x}``:

``\dfrac{∂f}{∂x}=\mathrm{pushforward}_{h(b)|b=g(x)}\left(\left.\dfrac{∂g}{∂a}\right|_{a=x}\right)``


### The anatomy of pullback and pushforward

For our function `foo(args...; kwargs...) = y`:


```julia
function pullback(Δy)
    ...
    return ∂self, ∂args...
end
```

The input to the pullback is often called the _seed_.
If the function is `y = f(x)` often the pullback will be written `s̄elf, x̄ = pullback(ȳ)`.

!!! note

    The pullback returns one `∂arg` per `arg` to the original function, plus one `∂self` for the fields of the function itself (explained below).

!!! terminology "perturbation, seed, sensitivity"
    Sometimes _perturbation_, _seed_, and even _sensitivity_ will be used interchangeably.
    They are not generally synonymous, and ChainRules shouldn't mix them up.
    One must be careful when reading literature.
    At the end of the day, they are all _wiggles_ or _wobbles_.


The pushforward is a part of the `frule` function.
Considered alone it would look like:

```julia
function pushforward(Δself, Δargs...)
    ...
    return ∂y
end
```
But because it is fused into frule we see it as part of:
```julia
function frule(::typeof(foo), args..., Δself, Δargs...; kwargs...)
    ...
    return y, ∂y
end
```


The input to the pushforward is often called the _perturbation_.
If the function is `y = f(x)` often the pushforward will be written `ẏ = last(frule(f, x, ṡelf, ẋ))`.
`ẏ` is commonly used to represent the perturbation for `y`.

!!! note

    In the `frule`/pushforward,
    there is one `Δarg` per `arg` to the original function.
    The `Δargs` are similar in type/structure to the corresponding inputs `args` (`Δself` is explained below).
    The `∂y` are similar in type/structure to the original function's output `Y`.
    In particular if that function returned a tuple then `∂y` will be a tuple of the same size.

### Self derivative `Δself`, `∂self`, `s̄elf`, `ṡelf` etc.

!!! terminology "Δself, ∂self, s̄elf, ṡelf"
    It is the derivatives with respect to the internal fields of the function.
    To the best of our knowledge there is no standard terminology for this.
    Other good names might be `Δinternal`/`∂internal`.

From the mathematical perspective, one may have been wondering what all this `Δself`, `∂self` is.
Given that a function with two inputs, say `f(a, b)`, only has two partial derivatives:
``\dfrac{∂f}{∂a}``, ``\dfrac{∂f}{∂b}``.
Why then does a `pushforward` take in this extra `Δself`, and why does a `pullback` return this extra `∂self`?

The reason is that in Julia the function `f` may itself have internal fields.
For example a closure has the fields it closes over; a callable object (i.e. a functor) like a `Flux.Dense` has the fields of that object.

**Thus every function is treated as having the extra implicit argument `self`, which captures those fields.**
So every `pushforward` takes in an extra argument, which is ignored unless the original function has fields.
It is common to write `function foo_pushforward(_, Δargs...)` in the case when `foo` does not have fields.
Similarly every `pullback` returns an extra `∂self`, which for things without fields is the constant `NO_FIELDS`, indicating there are no fields within the function itself.


### Pushforward / Pullback summary

- **Pullback**
   - returned by `rrule`
   - takes output space wobbles, gives input space wiggles
   - 1 argument per original function return
   - 1 return per original function argument + 1 for the function itself

- **Pushforward:**
    - part of `frule`
    - takes input space wiggles, gives output space wobbles
    - 1 argument per original function argument + 1 for the function itself
    - 1 return per original function return


### Pullback/Pushforward and Directional Derivative/Gradient

The most trivial use of the `pushforward` from within `frule` is to calculate the directional derivative:

If we would like to know the the directional derivative of `f` for an input change of `(1.5, 0.4, -1)`

```julia
direction = (1.5, 0.4, -1) # (ȧ, ḃ, ċ)
y, ẏ = frule(f, a, b, c, Zero(), direction)
```

On the basis directions one gets the partial derivatives of `y`:
```julia
y, ∂y_∂a = frule(f, a, b, c, Zero(), 1, 0, 0)
y, ∂y_∂b = frule(f, a, b, c, Zero(), 0, 1, 0)
y, ∂y_∂c = frule(f, a, b, c, Zero(), 0, 0, 1)
```

Similarly, the most trivial use of `rrule` and returned `pullback` is to calculate the [Gradient](https://en.wikipedia.org/wiki/Gradient):

```julia
y, f_pullback = rrule(f, a, b, c)
∇f = f_pullback(1)  # for appropriate `1`-like seed.
s̄elf, ā, b̄, c̄ = ∇f
```
Then we have that `∇f` is the _gradient_ of `f` at `(a, b, c)`.
And we thus have the partial derivatives ``\overline{\mathrm{self}}, = \dfrac{∂f}{∂\mathrm{self}}``, ``\overline{a} = \dfrac{∂f}{∂a}``, ``\overline{b} = \dfrac{∂f}{∂b}``, ``\overline{c} = \dfrac{∂f}{∂c}``, including the and the self-partial derivative, ``\overline{\mathrm{self}}``.

## Differentials

The values that come back from pullbacks or pushforwards are not always the same type as the input/outputs of the primal function.
They are differentials, which correspond roughly to something able to represent the difference between two values of the primal types.
A differential might be such a regular type, like a `Number`, or a `Matrix`, matching to the original type;
or it might be one of the `AbstractDifferential` subtypes.

Differentials support a number of operations.
Most importantly: `+` and `*`, which let them act as mathematical objects.

The most important `AbstractDifferential`s when getting started are the ones about avoiding work:

 - `Thunk`: this is a deferred computation. A thunk is a [word for a zero argument closure](https://en.wikipedia.org/wiki/Thunk). A computation wrapped in a `@thunk` doesn't get evaluated until `unthunk` is called on the thunk. `unthunk` is a no-op on non-thunked inputs.
 - `One`, `Zero`: There are special representations of `1` and `0`. They do great things around avoiding expanding `Thunks` in multiplication and (for `Zero`) addition.

### Other `AbstractDifferential`s:
 - `Composite{P}`: this is the differential for tuples and  structs. Use it like a `Tuple` or `NamedTuple`. The type parameter `P` is for the primal type.
 - `DoesNotExist`: Zero-like, represents that the operation on this input is not differentiable. Its primal type is normally `Integer` or `Bool`.
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
_, _, ā = b_pullback(extern(b̄));  # ∂c/∂a
_, x̄ = a_pullback(extern(ā));     # ∂c/∂x = ∂f/∂x
extern(x̄)
# -2.0638950738662625

#### Find dfoo/dx via frules

x = 3;
ẋ = 1;  # ∂x/∂x
nofields = Zero();  # ∂self/∂self

a, ȧ = frule(sin, x, nofields, ẋ); # ∂a/∂x
b, ḃ = frule(*, 2, nofields, unthunk(ȧ)); # ∂b/∂x = ∂b/∂a⋅∂a/∂x

c, ċ = frule(asin, b, unthunk(ḃ)); # ∂c/∂x = ∂c/∂b⋅∂b/∂x = ∂f/∂x
unthunk(ċ)
# -2.0638950738662625

#### Find dfoo/dx via finite-differences

using FiniteDifferences
central_fdm(5, 1)(foo, x)
# -2.0638950738670734

#### Find dfoo/dx via ForwardDiff.jl
using ForwardDiff
ForwardDiff.derivative(foo, x)
# -2.0638950738662625

#### Find dfoo/dx via Zygote.jl
using Zygote
Zygote.gradient(foo, x)
# (-2.0638950738662625,)
```
