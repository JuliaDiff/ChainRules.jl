```@meta
DocTestSetup = :(using ChainRulesCore, ChainRules)
```

# ChainRules

[ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) provides a variety of common utilities that can be used by downstream automatic differentiation (AD) tools to define and execute forward-, reverse-, and mixed-mode primitives.

This package is a work-in-progress, as is the documentation. Contributions welcome!

## TODO Include the following:
* rrule:
* frule:
* Pullback:  takes a Wobble in the output space, and tells you how much Wiggle you need to make in the input space to get that.
* Pushforward:  takes a Wibble in the input space,
* and tells you how much Wobble you get in the output space.
* Total derivative
* Gradient
* Seed
* Partial
* Permutation
* Sensitivity
* Thunk
* Differential
* Self-derivative, Internal derivative:



Note: The following terminology is for ChainRules purposes.
It should align with uses in general.
Be warned that differential geometers might make sad-faces when they realize ChainRule’s pullback / pushforwards are only for the very boring euclidean spaces.




### `rrule` and `frule`
ChainRules is all about providing a rich set of rules for doing differentiation.
When a person does introductory calculus, they learn that the derivative (with respect to `x`)
of `a*x` is `a`, and the derivative of `sin(x)` is `cos(x)`, etc.
And they learn how to combine simple rules, via the chain rule, to differentiate complicated functions.
ChainRules.jl basically a progamatic repository of that knowledge, with the generalizations to higher dimensions.

Autodiff (AD) tools roughly work by reducting a program down to simple parts that they know the rules for,
and then combining those rules.
Knowing rules for more complicated functions speeds up the autodiff process as it doesn't have to break things down as much.





________________




On writing good rrule / frules


* Use thunks appropriately:
   * If work is only required for 1 of the returned differentials it should be wrapped in a `@thunk` (potentially using a begin-end block)
   * If there are multiple return values, almost always their should be computation wrapped in a `@thunk`s


   * Don’t wrap variables in thunks, wrap the computations that fill those variables in thunks: Eg:
Write:
```
∂A = @thunk(foo(x))
return ∂A
```
                Not:
```
∂A = foo(x)
return @thunk(∂A)
```
In the bad example `foo(x)` gets computed eagerly, and all that the thunk is doing is wrapping the already calculated result in a function that returns it.




*  Style: used named local functions for the pushforward/pullback:
Rather than:
```
function frule(::typeof(foo), x)
        return (foo(x), (_, ẋ)->bar(ẋ))
end
```


write:


```
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
