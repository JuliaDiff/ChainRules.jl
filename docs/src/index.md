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

**ChainRules is an AD-independent collection of rules to use in a differentiation system.**

!!! note "The whole field is a mess for terminology"
    It isn't just ChainRules, it is everyone.
    Internally ChainRules tries to be consistent.
    Help with that is always welcomed.

### `frule` and `rrule`

!!! terminology "`frule` and `rrule`"
    `frule` and `rrule` are ChainRules specific terms.
    Their exact functioning is fairly ChainRules specific, though other tools have similar functions.
    The core notion is sometimes called _custom AD primitives_, _custom adjoints_, _custom_gradients_, _custom sensitivities_.

The rules are encoded as `frule`s and `rrule`s, for use in forward-mode and reverse-mode differentiation respectively.

The `frule` is written:
```julia
function frule(::typeof(foo), args; kwargs...)
    ...
    return y, pushforward
end
```
where `y = foo(args; kwargs...)`, and `pushforward` is a function to propagate the derivative information forwards at that point (more later).



The `rrule` for some function `foo`, which takes the positional argument `args` and keyword argument `kwargs`, is written:

```julia
function rrule(::typeof(foo), args; kwargs...)
    ...
    return y, pullback
end
```
again `y` must be equal to `foo(args; kwargs...)`, and `pullback` is a function to propagate the derivative information backwards at that point (more later).


Almost always the _pushforward_/_pullback_ will be declared locally within the `frule`/`rrule`, and will be a _closure_ over some of the other arguments.

### The propagators: pushforward and pullback

!!! terminology "pushforward and pullback"

    _Pushforward_ and _pullback_ are fancy words that the autodiff community adopted from Differential Geometry.
    The are broadly in agreement with the use of [pullback](https://en.wikipedia.org/wiki/Pullback_(differential_geometry)) and [pushforward](https://en.wikipedia.org/wiki/Pushforward_(differential)) in differential geometry.
    But any geometer will tell you these are the super-boring flat cases. Some will also frown at you.
    They are also sometimes described in terms of the jacobian:
    The _pushforward_ is _jacobian vector product_ (`jvp`), and _pullback_ is _jacobian transpose vector product_ (`j'vp`).
    Other terms that may be used include for _pullback_ the **backpropagator**, and by analogy for _pushforward_ the **forwardpropagator**, thus these are the _propagators_.
    These are also good names because effectively they propagate wiggles and wobbles through them, via the chain rule.
    (the term **backpropagator** may originate with ["Lambda The Ultimate Backpropagator"](http://www-bcl.cs.may.ie/~barak/papers/toplas-reverse.pdf) by Pearlmutter and Siskind, 2008)

#### Core Idea

##### Less formally

 - The **pushforward** takes a wiggle in the _input space_, and tells what wobble you would create in the output space, by passing it through the function.
 - The **pullback** takes wobblyness information with respect to the function's output, and tells the equivalent wobblyness with repect to the functions input.

##### More formally
The **pushforward** of ``f`` takes the _sensitivity_ of the input of ``f`` to a quantity, and gives the _sensitivity_ of the output of ``f`` to that quantity
The **pullback** of ``f`` takes the _sensitivity_ of a quantity to the output of ``f``, and gives the _sensitivity_ of that quantity to the input of ``f``.

#### Math
This is all a bit simplied by talking in 1D.

##### Lighter Math
For a chain of expressions:
```
a = f(x)
b = g(a)
c = h(b)
```

The pullback of `g`, which incorperates the knowledge of `∂b/∂a`,
applies the chainrule to go from `∂c/∂b` to `∂c/∂a`.

the pushforward of `g`,  which also incorperates the knowledge of `∂b/∂a`,
applies the chainrule to go from `∂a/∂x` to `∂b/∂x`.

#### Heavier Math
If I have some functions: ``g(a)``, ``h(b)`` and ``f(x)=g(h(x))``,
and I know the pullback of ``g``, at ``h(x)`` written: ``\mathrm{pullback}_{g(a)|a=h(x)}``,

and I know the deriviative of h with respect to its input ``b`` at ``g(x)``, written:
``\left.\dfrac{∂h}{∂b}\right|_{b=g(x)}``

Then I can use the pullback to find: ``\dfrac{∂f}{∂x}``

``\dfrac{∂f}{∂x}=\mathrm{\mathrm{pullback}_{g(a)|a=h(x)}}\left(\left.\dfrac{∂h}{∂b}\right|_{b=g(x)}\right)``

—

If I know the deriviative of g with respect to its input a at x, written: ``\left.\dfrac{∂g}{∂a}\right|_{a=x}``

and I know the pushforward of ``h`` at ``g(x)`` written: ``\mathrm{pushforward}_{h(b)|b=g(x)}``

then I can use the pushforward to find ``\dfrac{∂f}{∂x}``

``\dfrac{∂f}{∂x}=\mathrm{pushforward}_{h(b)|b=g(x)}\left(\left.\dfrac{∂g}{∂a}\right|_{a=x}\right)``


#### The anatomy of pushforward and pullback

For our function `foo(args...; kwargs) = Y`:

The pushforward is a function:

```julia
function pushforward(Δself, Δargs...)
    ...
    return ∂Y
end
```

The input to the pushforward is often called the _perturbation_.
If the function is `y = f(x)` often the pushforward will be written `ẏ = pushforward(ṡelf, ẋ)`.
(`ẏ` is commonly used to represent the pertubation for `y`)

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
If the function is `y = f(x)` often the pullback will be written `s̄elf, x̄ = pullback(ȳ)`.

!!! note

    The pullback returns one `∂arg` per `arg` to the original function, plus one for the fields of the function itself (explained below).

!!! terminology
    Sometimes _perturbation_, _seed_, and even _sensitivity_ will be used interchangeably.
    They are not generally synonymous, and ChainRules shouldn't mix them up.
    One must be careful when reading literature.
    At the end of the day, they are all _wiggles_ or _wobbles_.

### Self derivative `Δself`, `∂self`, `s̄elf`, `ṡelf` etc.

!!! terminology  `Δself`, `∂self`, `s̄elf`, `ṡelf`
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

#### Pushforward / Pullback summary

- **Pushforward:**
    - returned by `frule`
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
s̄elf, ā, b̄, c̄ = ∇f
```
Then we have that `∇f` is the _gradient_ of `f` at `(a, b, c)`.
And we thus have the partial derivatives ``\overline{\mathrm{self}}, = \dfrac{∂f}{∂\mathrm{self}}``, ``\overline{a} = \dfrac{∂f}{∂a}``, ``\overline{b} = \dfrac{∂f}{∂b}``, ``\overline{c} = \dfrac{∂f}{∂c}``, including the and the self-partial derivative, ``\overline{\mathrm{self}}``.

### Differentials

The values that come back from pullbacks or pushforwards are not always the same type as the input/outputs of the original function.
They are differentials, which correspond roughly to something able to represent the difference between two values of the original types.
A differential might be such a regular type, like a `Number`, or a `Matrix`, matching to the original type;
or it might be one of the `AbstractDifferential` subtypes.

Differentials support a number of operations.
Most importantly: `+` and `*`, which let them act as mathematical objects.
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

#### Find dfoo/dx via ForwardDiff.jl
using ForwardDiff
ForwardDiff.derivative(foo, x)
# -2.0638950738662625

#### Find dfoo/dx via Zygote.jl
using Zygote
Zygote.gradient(foo, x)
# (-2.0638950738662625,)
```

 -------------------------------

## On writing good `rrule` / `frule` methods

### Use `Zero()` or `One()` as return value

The `Zero()` and `One()` differential objects exist as an alternative to directly returning
`0` or `zeros(n)`, and `1` or `I`.
They allow more optimal computation when chaining pullbacks/pushforwards, to avoid work.
They should be used where possible.

### Use `Thunk`s appropriately:

If work is only required for one of the returned differentials, then it should be wrapped in a `@thunk` (potentially using a `begin`-`end` block).

If there are multiple return values, their computation should almost always be wrapped in a `@thunk`.

Do _not_ wrap _variables_ in a `@thunk`; wrap the _computations_ that fill those variables in `@thunk`:

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
Instead you probably want `transpose(a)`, unless you've already restricted `a` to be a `AbstractMatrix{<:Real}`.

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
#== output
julia> frule(foo, 2)
(4, var"#foo_pushforward#11"())
==#

# bad:
function frule(::typeof(foo), x)
    return foo(x), (_, ẋ) -> bar(ẋ)
end
#== output:
julia> frule(foo, 2)
(4, var"##9#10"())
==#
```

While this is more verbose, it ensures that if an error is thrown during the `pullback`/`pushforward` the [`gensym`](https://docs.julialang.org/en/v1/base/base/#Base.gensym) name of the local function will include the name you gave it.
This makes it a lot simpler to debug from the stacktrace.

### Write tests

There are fairly decent tools for writing tests based on [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl).
They are in [`tests/test_utils.jl`](https://github.com/JuliaDiff/ChainRules.jl/blob/master/test/test_util.jl).
Take a look at existing test and you should see how to do stuff.

!!! warning
    Use finite differencing to test derivatives.
    Don't use analytical derivations for derivatives in the tests.
    Those are what you use to define the rules, and so can not be confidently used in the test.
    If you misread/misunderstood them, then your tests/implementation will have the same mistake.

### CAS systems are your friends.

It is very easy to check gradients or derivatives with a computer algebra system (CAS) like [WolframAlpha](https://www.wolframalpha.com/input/?i=gradient+atan2%28x%2Cy%29).

------------------------------------------

## FAQ

### What is up with the different symbols?

#### `Δx`, `∂x`, `dx`
ChainRules uses these perhaps atyptically.
As a notation that is the same across propagators, regardless of direction. (Incontrast see `ẋ` and `x̄` below)

 - `Δx` is the input to a propagator, (i.e a _seed_ for a _pullback_; or a _perturbation_ for a _pushforward_)
 - `∂x` is the output of a propagator
 - `dx` could be anything, including a pullback/pushforward. It really should not show up outside of tests.


#### ``\dot{y} = \dfrac{∂y}{∂x} = \overbar{x}``
 - `v̇` is a derivative of the input moving forward: ``v̇ = \frac{∂v}{∂x}`` for input ``x``, intermediate value ``v``.
 - `v̄` is a derivative of the output moving backward: ``v̄ = \frac{∂y}{∂v}`` for output ``y``, intermediate value ``v``.

#### others
 - `Ω` is often used as the return value of the function. Especially, but not exclusively, for scalar functions.
     - `ΔΩ` is thus a seed for the pullback.
     - `∂Ω` is thus the output of a pushforward.


### Why does `frule` and `rrule` return the function evaluation?
You might wonder why `frule(f, x)` returns `f(x)` and the pushforward for `f` at `x`, and similarly for `rrule` returing `f(x)` and the pullback for `f` at `x`.
Why not just return the pushforward/pullback, and let the user call `f(x)` to get the answer seperately?

There are two reasons the rules also calculate the `f(x)`.
1. For some rules the output value is used in the definition of its propagator. For example `tan`.
2. For some rules an alternative way of calculating `f(x)` can give the same answer while also generating intermediate values that can be used in the calculations within the propagator.

### Where are the gradients for keyword arguments?
_pullbacks_ do not return a gradient for keyword arguments;
similarly _pushfowards_ do not accept a pertubation for keyword arguments.
This is because in practice functions are very rarely differentiable with respect to keyword arguments.
As a rule keyword arguments tend to control side-effects, like logging verbsoity,
or to be functionality changing to perform a different operation, e.g. `dims=3`, and thus not differentiable.
To the best of our knowledge no julia AD system, with support for the definition of custom primatives, supports differentating with respect to keyword arguments.
At some point in the future ChainRules may support these. Maybe.
