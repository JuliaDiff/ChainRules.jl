# On writing good `rrule` / `frule` methods

## Use `Zero()` or `One()` as return value

The `Zero()` and `One()` differential objects exist as an alternative to directly returning
`0` or `zeros(n)`, and `1` or `I`.
They allow more optimal computation when chaining pullbacks/pushforwards, to avoid work.
They should be used where possible.

## Use `Thunk`s appropriately

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

## Be careful with using `adjoint` when you mean `transpose`

Remember for complex numbers `a'` (i.e. `adjoint(a)`) takes the complex conjugate.
Instead you probably want `transpose(a)`, unless you've already restricted `a` to be a `AbstractMatrix{<:Real}`.

## Code Style

Use named local functions for the `pushforward`/`pullback`:

```julia
# good:
function frule(::typeof(foo), x)
    Y = foo(x)
    function foo_pushforward(_, ẋ)
        return bar(ẋ)
    end
    return Y, foo_pushforward
end
#== output
julia> frule(foo, 2)
(4, var"#foo_pushforward#11"())
==#

# bad:
function frule(::typeof(foo), x)
    return foo(x), (_, ẋ) -> bar(ẋ)
end
#== output:
julia> frule(foo, 2)
(4, var"##9#10"())
==#
```

While this is more verbose, it ensures that if an error is thrown during the `pullback`/`pushforward` the [`gensym`](https://docs.julialang.org/en/v1/base/base/#Base.gensym) name of the local function will include the name you gave it.
This makes it a lot simpler to debug from the stacktrace.

## Write tests

There are fairly decent tools for writing tests based on [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl).
They are in [`tests/test_utils.jl`](https://github.com/JuliaDiff/ChainRules.jl/blob/master/test/test_util.jl).
Take a look at existing test and you should see how to do stuff.

!!! warning
    Use finite differencing to test derivatives.
    Don't use analytical derivations for derivatives in the tests.
    Those are what you use to define the rules, and so can not be confidently used in the test.
    If you misread/misunderstood them, then your tests/implementation will have the same mistake.

## CAS systems are your friends.

It is very easy to check gradients or derivatives with a computer algebra system (CAS) like [WolframAlpha](https://www.wolframalpha.com/input/?i=gradient+atan2%28x%2Cy%29).
