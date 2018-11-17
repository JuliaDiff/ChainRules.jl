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
##### `@forward_rule`
#####

"""
    @forward_rule(signature, f(x, y, ...),
                  (df₁_dx, df₁_dy, ...),
                  (df₂_dx, df₂_dy, ...),
                  ...)

Define a method for `forward_rule` using the given signature, call expression,
and derivative expressions.

While this macro is convenient for avoiding boilerplate code when implementing
simple forward rules, note that more advanced rules will probably require
overloading `ChainRules.forward_rule` directly. For now, the macro only
supports real-domain rules.

Examples:

`@forward_rule(R → R, sin(x), cos(x))` expands to:

    function forward_rule(::@sig(R → R), ::typeof(sin), x)
        return sin(x), ẋ -> forward_chain(@thunk(cos(x)), ẋ)
    end

`@forward_rule(R⊕R → R, *(x, y), (y, x))` expands to:

    function forward_rule(::@sig(R → R), ::typeof(*), x, y)
        return *(x, y), (ẋ, ẏ) -> forward_chain(@thunk(y), ẋ, @thunk(x), ẏ)
    end

`@forward_rule(R⊕R → R, sincos(x), cos(x), -sin(x))` expands to:

    function forward_rule(::@sig(R → R⊕R), ::typeof(sincos), x)
        return sincos(x),
               (ẋ -> forward_chain(ẋ, @thunk(cos(x))),
                ẋ -> forward_chain(ẋ, @thunk(-sin(x))))
    end

Note that this last case is a good example of a primitive that is more
efficiently implemented with a manual `forward_rule` overload:

    function forward_rule(::@sig(R → R⊕R), ::typeof(sincos), x)
        sinx, cosx = sincos(x)
        return (sinx, cosx),
               (ẋ -> forward_chain(ẋ, @thunk(cosx)),
                ẋ -> forward_chain(ẋ, @thunk(-sinx)))
    end
"""
macro forward_rule(signature, call, derivs...)
    return generate_rule_definition(signature, call, nothing, derivs...)
end

"""
    @reverse_rule(signature, f(x, y, ...), (f̄₁, f̄₂, ...), df_dx, df_dy, ...)

Define a method for `reverse_rule` using the given signature, call expression,
and derivative expressions.

The third argument to the macro is a tuple of symbols naming the adjoints of the
outputs of `f(x, y, ...)` (or just a single symbol if there is only one output).

While this macro is convenient for avoiding boilerplate code when implementing
simple reverse rules, note that more advanced rules will probably require
overloading `ChainRules.reverse_rule` directly. For now, this macro only
supports real-domain rules.

Examples:

`@reverse_rule([R] → R, sum(x), ȳ, ȳ)` expands to:

    function reverse_rule(::@sig([R] → R), ::typeof(sum), x)
        return sum(x), (x̄, ȳ) -> reverse_chain!(x̄, @thunk(ȳ))
    end

`@reverse_rule([R]⊕[R] → [R], *(x, y), z̄, z̄ * y', x' * z̄)` expands to:

    function reverse_rule(::@sig([R]⊕[R] → R), ::typeof(*), x, y)
        return x * y,
               ((x̄, z̄) -> reverse_chain!(x̄, @thunk(z̄ * y')),
                (ȳ, z̄) -> reverse_chain!(ȳ, @thunk(x' * z̄)))
    end
"""
macro reverse_rule(signature, call, adjoint_names, derivs...)
    if Meta.isexpr(adjoint_names, :tuple)
        adjoint_names = convert(Vector{Symbol}, adjoint_names.args)
    elseif isa(adjoint_names, Symbol)
        adjoint_names = Symbol[adjoint_names]
    end
    return generate_rule_definition(signature, call, adjoint_names, derivs...)
end

# TODO: Expand this beyond real-domain rules by parsing signature ahead of time
function generate_rule_definition(signature, call,
                                  adjoint_names::Union{Nothing,Vector{Symbol}},
                                  derivs...)
    @assert Meta.isexpr(call, :call)
    call_function = esc(call.args[1])
    call_args = map(esc, call.args[2:end])
    seed_names = Any[Symbol(string(:seed_, i)) for i in 1:length(call_args)]
    chains = Any[]
    if isa(adjoint_names, Nothing) # we're doing forward mode
        for deriv in derivs
            thunkables = Meta.isexpr(deriv, :tuple) ? deriv.args : [deriv]
            thunks = [:(@thunk($(esc(t)))) for t in thunkables]
            @assert length(thunks) == length(call_args)
            chain_call = Expr(:call, :forward_chain)
            for i in 1:length(thunks)
                push!(chain_call.args, seed_names[i])
                push!(chain_call.args, thunks[i])
            end
            push!(chains, :($(Expr(:tuple, seed_names...)) -> $chain_call))
        end
    else # we're doing reverse mode
        @assert length(derivs) == length(call_args)
        @assert length(adjoint_names) > 0
        adjoint_names = map(esc, adjoint_names)
        for i in 1:length(derivs)
            seed_name = seed_names[i]
            chain_call = Expr(:call, :reverse_chain!)
            push!(chain_call.args, seed_name)
            push!(chain_call.args, :(@thunk($(esc(derivs[i])))))
            push!(chains, :($(Expr(:tuple, seed_name, adjoint_names...)) -> $chain_call))
        end
    end
    @assert length(chains) > 0
    chains = length(chains) > 1 ? Expr(:tuple, chains...) : chains[1]
    rule_function = isa(adjoint_names, Nothing) ? :forward_rule : :reverse_rule
    return quote
        @assert(@sig($signature) <: Signature{<:Tuple{Vararg{Union{RealTensor,RealScalar,Ignore}}},
                                              <:Tuple{Vararg{Union{RealTensor,RealScalar,Ignore}}}},
                "@forward_rule and @reverse_rule only support real-domain rules right now")
        function $ChainRules.$rule_function(::@sig($signature),
                                            ::typeof($call_function),
                                            $(call_args...))
            outputs = $(esc(call))
            return outputs, $chains
        end
    end
end

#####
##### forward rules
#####

@forward_rule(R → R, sin(x), cos(x))
@forward_rule(R → R, cos(x), -sin(x))
@forward_rule(R → R, log(x), inv(x))
@forward_rule(R⊕R → R, *(x, y), (y, x))

function forward_rule(::@sig(R⊕R → R), ::typeof(atan), y, x)
    h = hypot(y, x)
    return atan(y, x), (ẏ, ẋ) -> forward_chain(ẏ, @thunk(x / h), ẋ, @thunk(y / h))
end

function forward_rule(::@sig(R⊕R → R), ::typeof(hypot), x, y)
    h = hypot(x, y)
    return h, (ẋ, ẏ) -> forward_chain(ẋ, @thunk(x / h), ẏ, @thunk(y / h))
end

function forward_rule(::@sig(R → R⊕R), ::typeof(sincos), x)
    sinx, cosx = sincos(x)
    return (sinx, cosx),
           (ẋ -> forward_chain(ẋ, @thunk(cosx)),
            ẋ -> forward_chain(ẋ, @thunk(-sinx)))
end

forward_rule(::@sig(C → C), ::typeof(conj), x) = conj(x), ẋ -> (false, true)

#####
##### reverse rules
#####

@reverse_rule([R] → R, sum(x), ȳ, ȳ)
@reverse_rule([R]⊕[R] → [R], +(x, y), z̄, z̄, z̄)
@reverse_rule([R]⊕[R] → [R], *(x, y), z̄, z̄ * y', x' * z̄)

# TODO: This partial derivative extraction should be doable without the extra
# temporaries or preallocation utilized here, but AFAICT such an approach is
# hard to write without relying on inference hacks unless we have something
# akin to https://github.com/JuliaLang/julia/issues/22129
function reverse_rule(::@sig(_⊕[R] → [R]), ::typeof(broadcast), f, x)
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
