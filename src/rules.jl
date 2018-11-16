#####
##### fallback rules
#####

#=
There's an idea at play here that's not made explicit by these default "no
primitive available" fallback implementations.

In some weird ideal sense, the fallback for e.g. `forward_rule` should actually
be "get the derivative via forward-mode AD". This is necessary to enable
mixed-mode rules, where e.g.  `forward_rule` is used within a `reverse_rule`
definition. For example, broadcasted functions may not themselves be
forward-mode *primitives*, but are often forward-mode *differentiable*.

ChainRules, by design, is decoupled from any specific AD implementation. How,
then, do we know which AD to fall back to when there isn't a primitive defined?

Well, if you're a greedy AD implementation, you can just overload `forward_rule`
and/or `reverse_rule` to use your AD directly. However, this won't place nice
with other AD packages doing the same thing, and thus could cause
load-order-dependent problems for downstream users.

It turns out, Cassette solves this problem nicely by allowing AD authors to
overload the fallbacks w.r.t. their own context. Example using ForwardDiff:

```
using ChainRules, ForwardDiff, Cassette

Cassette.@context MyChainRuleCtx

# ForwardDiff, itself, can call `my_forward_rule` instead of
# `forward_rule` to utilize the ForwardDiff-injected ChainRules
# infrastructure
my_forward_rule(args...) = Cassette.overdub(MyChainRuleCtx(), forward_rule, args...)

function Cassette.execute(::MyChainRuleCtx, ::typeof(forward_rule)
                          ::@sig(R → R), f, x)
    fx, df = forward_rule(sig, f, x)
    if isa(df, Nothing)
        fx, df = (f(x), ẋ -> ẋ * ForwardDiff.derivative(f, x))
    end
    return fx, df
end
```
=#

forward_rule(::Signature, ::Vararg{Any}) = (nothing, nothing)

reverse_rule(::Signature, ::Vararg{Any}) = (nothing, nothing)

# TODO: Should the default be to whitelist known holomorphic functions, or to
# blacklist known non-holomorphic functions? This implements the latter.
function forward_rule(signature::@sig(C → C), f, x)
    fx, df = forward_rule(Signature(RealScalar(), RealScalar()), f, x)
    return fx, ẋ -> (df(ẋ), false)
end

function reverse_rule(signature::@sig(R → R), f, x)
    fx, df = forward_rule(signature, f, x)
    return fx, (x̄, z̄) -> reverse_chain!(x̄, @thunk(df(z̄)))
end

#####
##### forward rules
#####

function forward_rule(::@sig(R → R), ::typeof(sin), x)
    return sin(x), ẋ -> forward_chain(@thunk(cos(x)), ẋ)
end

function forward_rule(::@sig(R → R), ::typeof(cos), x)
    return cos(x), ẋ -> forward_chain(@thunk(-sin(x)), ẋ)
end

function forward_rule(::@sig(R → R⊗R), ::typeof(sincos), x)
    sinx, cosx = sincos(x)
    return sinx, ẋ -> forward_chain(@thunk(cosx), ẋ),
           cosx, ẋ -> forward_chain(@thunk(-sinx), ẋ)
end

function forward_rule(::@sig(R⊗R → R), ::typeof(atan), y, x)
    h = hypot(y, x)
    return atan(y, x),
           (ẏ, ẋ) -> forward_chain(@thunk(x / h), ẏ, @thunk(y / h), ẋ)
end

function forward_rule(::@sig(R → R), ::typeof(log), x)
    return log(x), ẋ -> forward_chain(@thunk(inv(x)), ẋ)
end

forward_rule(::@sig(C → C), ::typeof(conj), x) = conj(x), ẋ -> (false, true)

#####
##### reverse rules
#####

function reverse_rule(::@sig([R] → R), ::typeof(sum), x)
    return sum(x),
           (x̄, z̄) -> reverse_chain!(x̄, @thunk(z̄))
end

function reverse_rule(::@sig([R]⊗[R] → R), ::typeof(+), x, y)
    return x + y,
           (x̄, ȳ, z̄) -> (reverse_chain!(x̄, @thunk(z̄)),
                         reverse_chain!(ȳ, @thunk(z̄)))
end

function reverse_rule(::@sig([R]⊗[R] → R), ::typeof(*), x, y)
    return x * y,
           (x̄, ȳ, z̄) -> (reverse_chain!(x̄, @thunk(z̄ * y')),
                         reverse_chain!(ȳ, @thunk(x' * z̄)))
end

# TODO: This partial derivative extraction should be doable without the extra
# temporaries or preallocation utilized here, but AFAICT such an approach is
# hard to write without relying on inference hacks unless we have something
# akin to https://github.com/JuliaLang/julia/issues/22129
function reverse_rule(::@sig(_⊗[R] → [R]), ::typeof(broadcast), f, x)
    s = Signature(RealScalar(), RealScalar())
    f_rule = x -> begin
        y, d = forward_rule(s, f, x)
        y, d(one(x))
    end
    applied_f_rule = broadcast(f_rule, x)
    values = map(first, applied_f_rule)
    derivs = map(last, applied_f_rule)
    return values, (x̄, z̄) -> reverse_chain!(x̄, @thunk(broadcasted(*, z̄, derivs)))
end
