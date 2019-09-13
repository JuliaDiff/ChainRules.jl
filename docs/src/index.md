```@meta
DocTestSetup = :(using ChainRulesCore, ChainRules)
```

# ChainRules

[ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) provides a variety of common utilities that can be used by downstream automatic differentiation (AD) tools to define and execute forward-, reverse-, and mixed-mode primitives.

### Introduction:
ChainRules is all about providing a rich set of rules for doing differentiation.
When a person does introductory calculus, they learn that the derivative (with respect to `x`)
of `a*x` is `a`, and the derivative of `sin(x)` is `cos(x)`, etc.
And they learn how to combine simple rules, via the chain rule, to differentiate complicated functions.
ChainRules.jl basically a progamatic repository of that knowledge, with the generalizations to higher dimensions.

Autodiff (AD) tools roughly work by reducing a problem down to simple parts that they know the rules for,
and then combining those rules.
Knowing rules for more complicated functions speeds up the autodiff process as it doesn't have to break things down as much.

** ChainRules is an AD independent collection of rules to use in an differentiation system **

### `rrule` and `frule`

!!! Terminology "`rrule` and `frule`"

    `rrule` and `frule` are ChainRules.jl specific terms.
    And there exact functioning is kind of ChainRule specific,
    though other tools may do similar.
    The core notion is sometimes called
    _Custom AD primitives_, _custom adjoints_, _custom sensitivities_.

The rules are encoded as `rrules` and `frules`,
for use in reverse-mode and forward-mode differentiation respectively.

The `rrule` for some function `foo`, takes the positional argument `args` and keyword argument `kwargs` is written:
```julia
function rrule(::typeof(foo), args; kwargs...)
    ...
    return y, pullback
end
```
where `y` must be equal to `foo(args; kwargs...)`,
and _pullback_ is a function to propagate the derivative information backwards at that point (more later).
Often but not always it is calculated directly.
the exeception is we can calculate it indirect to make
the `pullback` faster. (more on _pullback_ later)

Similarly, the `frule` is written:
```julia
function frule(::typeof(foo), args; kwargs...)
    ...
    return y, pushforward
end
```
again `y=foo(args, kwargs...)`,
and _pushforward_ is a function to propagate the derivative information forwards at that point (more later).

Almost always the _pushforward_/_pullback_ will be declared locally with-in the `ffrule`/`rrule`, and will be a _closure_ over some of the other arguments.

### The propagators: pushforward and pullback

!!! Terminology "Pushforward and Pullback"

    _Pushforward_ and _Pullback_ are fancy words that the autodiff community recently stole from Differential Geometry.
    The are broadly in agreement with the use of these terms in differential geometry. But any geometer will tell you these are the super-boring flat cases. Some will also frown at you.
    Other terms that may be used include for _pullback_ the **backpropagator**, and by analogy for _pushforward_ the **forwardpropagator**, thus these are the _propagators_.
    These are also good names because effectively they propagate wibbles and wobbles through them, via the chainrule.
    (the term **backpropagator** may originate with ["Lambda The Ultimate Backpropagator"](http://www-bcl.cs.may.ie/~barak/papers/toplas-reverse.pdf) by Bearlmutter and Siskind, 2008)


#### Core Important Idea:
 - The **Pushforward** takes a wiggle in the _input space_, and tells what wobble you would create in the output space, by passing it through the function.
 - The **Pullback** takes a wobble in the _output space_, and tells you what wiggle you would need to make in the _input_ space to achieve it.

#### The anatomy of pushforward and pullback

For our function `foo(args...; kwargs) = Y`:

The pushforward is a function:
```julia
function pushforward(Δself, Δargs...)
    ...
    return ∂Y
end
```
Note that there is one `Δargs...` per `arg` to the orginal function, and they are similar in type/structure to the ccorresponding inputs.
Plus the `Δself` (don't worry we will be back to explain this soon).
The `∂Y` will be similar in type/structure to the original function's output `Y`.
In particular if that function returned a tuple then `∂Y` will be a tuple of same size.

The input to the pushforward is often called the _pertubation_.
If the function is `y=f(x)` often the pushforward will be written `ẏ=pushforward(ẋ)`.


The pullback is a function
```julia
function pullback(ΔY)
    ...
    return ∂self, ∂args...
end
```

Note that the pullback returns one `∂arg` per original `arg` to the function, plus one for the s

The input to the pullback is often called the _seed_.
If the function is `y=f(x)` often the pullback will be written `ȳ=pullback(x̄)`.


!!! Terminology:
    Sometimes _pertubation_, _seed_, _sensitivity_ will be used interchangeably, depending on task/subfield (_sensitivity_ analysis and perturbation analysis are apparently very big on just calling everying _sensitivity_ or _pertubation_ respectively.)
    At the end of the day they are all _wibbles_ or _wobbles_.

### self derivative `Δself`, `∂self` etc.

!!! Terminology
    To my knowledge there is no standard termanology for this.
    Other good names might be `Δinternal`/`∂internal`

From the mathematical perspective,
one may have been wondering what all this `Δself`, `∂self` is.
After all a function with two inputs:
say `f(a, b)` only has two partial derivatives,
``\dfrac{∂f}{∂a}``, ``\dfrac{∂f}{∂b}``,
why then does the _pushforward_ take in this extra `Δself`,
and why does the _pullback_ return this extra `∂self` ?

The thing is in julia
the function `f` may itself have internal values.
For example a closure has the fields it closes over; and a callable object (i.e. a functor) like a `Flux.Dense` has the fields of that object.

**Thus every function is treated as having the extra implicit argument `self`,
which captures those fields.**
So all _pushforward_ take in a extra argument,
which unless they are for things with fields, they ignore. (thus common to write `function pushforward(_, Δargs...)` in those cases),
and every _pullback_ return an extra `∂self`,,
which is, for things without fields, the constant `NO_FIELDS` which indicates there is no fields within the function itself.


#### Pushforward / Pullback summary
- **Pushforward:**
    - returned by `ffrule`
    - takes input space wibbles, gives output space wobbles
    - 1 argument per orignal function argument + 1 for the function itself
    - 1 return per orignal function return
- **Pullback**
   -  return by `rrule`
   - takes output space wobbles, gives input space wibbles
   - 1 argument per original function return
   - 1 return per orignal function argument + 1 for the function itself

#### Pushforward/Pullback and Total Derivative/Gradient

The most trivial use of the frule+pushforward is to calculate the [Total Derivative](https://en.wikipedia.org/wiki/Total_derivative):
```julia
y, pushforward = frule(f, a, b, c)
ẏ = pushforward(1, 1, 1, 1)  # for appropriate `1`-like perturbation.
```
Then we have that
`ẏ` is the _total derivative_ of
`f` at `(a, b, c)`:
written mathematically as ``df_{(a,b,c)}``


Similarly:
The most trivial use of the rrule+pullback is to calculate the [Gradient](https://en.wikipedia.org/wiki/Gradient):
```julia
y, pullback = frule(f, a, b, c)
∇f  = pushforward(1) # for appropriate `1`-like seed.
s̄, ā, b̄, c̄ = ∇f
```
Then we have that
`∇f` is the _gradient_ of
`f` at `(a, b, c)`.
And we thus have the partial derivative:
s̄, ā, b̄, c̄.
(Including the and the self-partial derivative,
s̄).
Written mathematically as ``\dfrac{∂f}{∂a}``, ``\dfrac{∂f}{∂b}``, ``\dfrac{∂f}{∂c}``.


### Differentials

The values that come back from pullbacks,
or pushforwards
are not always the same type as the input/outputs of the original function.
They are differentials,
differency-equivalents.
A differential might be such a regular type,
like a Number, or a Matrix,
or it might be one of the `AbstractDifferencial` subtypes.

Differentials support a number of operations.
Most importantly:
`+` and `*` which lets them act as mathematically objects.
And `extern` which converts them into a conventional type.

The most important AbstractDifferentials when getting started are the ones about avoiding work:

 - `Thunk`: this is a deferred computation. A thunk is a [word for a zero argument closure](https://en.wikipedia.org/wiki/Thunk). An computation wrapped in a `@thunk` doesn't get evaluated until `extern` is called on the `Thunk`. More on thunks later.
 - `One`, `Zero`: There are special representions of `1` and `0`. They do great things around avoiding expanding `Thunks` in multiplication and (for `Zero`) addition.



#### Others: don't worry about them right now
 - Wirtinger: it is complex. The docs need to be better. [Read the links in this issue](https://github.com/JuliaDiff/ChainRulesCore.jl/issues/40).
 - Casted: it implements broadcasting mechanics. See [#10](https://github.com/JuliaDiff/ChainRulesCore.jl/issues/10)
 - InplacableThunk: it is like a Thunk but it can do `store!` and `accumulate!` inplace.


 -------------------------------
## Example of using ChainRules directly.

While ChainRules is largely intended as a backend for Autodiff systems it can be used directly.
(Infact this can be very useful if you can constraint the code you need to differnetiate to only use thing that have rules defined for.
This was once how all neural network code worked.)

Using ChainRules directly also helped get a feel for it.


```julia
using ChainRules

function foo(x)
    a = sin(x)
    b = 2a
    c = asin(b)
    return c
end;

###
# Find dfoo/dx via rrules

# First the forward pass, accumulating rules
x=3;
a, a_pb = rrule(sin, x);
b, b_pb = rrule(*, 2, a);
c, c_pb = rrule(asin, b)

# Then the backward pass calculating gradients
c̄ = 1;
_, b̄ = c_pb(extern(c̄));
_, _, ā = b_pb(extern(b̄));
_, x̄ = a_pb(extern(ā));
extern(x̄)
# -2.0638950738662625

###
# Find dfoo/dx via frules

# Unlike rrule can interleave evaluation and derivative evaluation
x=3;
ẋ=1;
nofields = NamedTuple();

a, a_pf = frule(sin, x);
ȧ = a_pf(nofields, extern(ẋ));

b, b_pf = frule(*, 2, a);
ḃ = b_pf(nofields, 0, extern(ȧ));

c, c_pf = frule(asin, b);
ċ = c_pf(nofields, extern(ḃ));
extern(ċ)
# -2.0638950738662625

###
# Find dfoo/dx via finite-difference
using FiniteDifferences
central_fdm(5,1)(foo, x)
# -2.0638950738670734

###
# Via ForwardDiff.jl
using ForwardDiff
ForwardDiff.derivative(foo, x)
# -2.0638950738662625

###
# Via Zygote
using Zygote
Zygote.gradient(foo, x)
# (-2.0638950738662625,)
```


 -------------------------------



## On writing good rrule / frules

### Return Zero or One
rather tan `0` or `1`
or even rather than `zeros(n)`, `ones(m,n)`

### Use thunks appropriately:

If work is only required for 1 of the returned differentials it should be wrapped in a `@thunk` (potentially using a begin-end block)

If there are multiple return values, almost always their should be computation wrapped in a `@thunk`s

Don’t wrap variables in thunks, wrap the computations that fill those variables in thunks: Eg:
Write:
```julia
∂A = @thunk(foo(x))
return ∂A
```
Not:
```julia
∂A = foo(x)
return @thunk(∂A)
```
In the bad example `foo(x)` gets computed eagerly, and all that the thunk is doing is wrapping the already calculated result in a function that returns it.

### Becareful of using Adjoing when you mean Transpose

Rember for complex numbers `a'` (i.e. `adjoint(a)`) takes the complex conjugate. Instead you probably want `transpose(a)`.

While there are arguments that for reverse-mode
taking the adjoint is correct, it is not currently the behavour of ChainRules to do so.
Feel free to open an issue and fight about it.
All differentials support `conj` efficiently, which makes it easy to change in post.

### Style

Used named local functions for the pushforward/pullback:
Rather than:
```julia
function frule(::typeof(foo), x)
        return (foo(x), (_, ẋ)->bar(ẋ))
end
```
Whichrite:
```julia
function frule(::typeof(foo), x)
        Y = foo(x)
        function foo_pushforward(_, ẋ)
            return bar(ẋ)
        end
        return Y, foo_pushforward
end
```


While this is more verbose,
it ensures that if an error is thrown during the pullback/pushforward
the gensymed name of the local function will include the name you gave it.
Which makes it a lot simpler to debug from the stacktrace.

### Write tests
There are faily decent tools for writing tests based on [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl)
They are in [`tests/test_utils.jl`](https://github.com/JuliaDiff/ChainRules.jl/blob/master/test/test_util.jl)
Take a look at existing test and you should see how to do stuff.

!!! important
    Don't write equations in tests.
    Use finite differencing.
    If you are writing equations in the tests, then you use those same equations as use are using to write your code. Then that is not Ok. We've had several bugs from people misreading/misunderstanding equations, and then using them for both tests and code. And then we have good coverage that is worthless.

### CAS systems are your friends.
E.g. it is very easy to check gradients or deriviatives with [WolframAlpha](https://www.wolframalpha.com/input/?i=gradient+atan2%28x%2Cy%29).
