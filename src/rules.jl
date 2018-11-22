#####
##### fallback rules
#####

#=
There's an idea at play here that's not made explicit by these default "no
primitive available" fallback implementations.

In some weird ideal sense, the fallback for e.g. `frule` should actually
be "get the derivative via forward-mode AD". This is necessary to enable
mixed-mode rules, where e.g.  `frule` is used within a `rrule`
definition. For example, broadcasted functions may not themselves be
forward-mode *primitives*, but are often forward-mode *differentiable*.

ChainRules, by design, is decoupled from any specific AD implementation. How,
then, do we know which AD to fall back to when there isn't a primitive defined?

Well, if you're a greedy AD implementation, you can just overload `frule`
and/or `rrule` to use your AD directly. However, this won't place nice
with other AD packages doing the same thing, and thus could cause
load-order-dependent problems for downstream users.

It turns out, Cassette solves this problem nicely by allowing AD authors to
overload the fallbacks w.r.t. their own context. Example using ForwardDiff:

```
using ChainRules, ForwardDiff, Cassette

Cassette.@context MyChainRuleCtx

# ForwardDiff, itself, can call `my_frule` instead of
# `frule` to utilize the ForwardDiff-injected ChainRules
# infrastructure
my_frule(args...) = Cassette.overdub(MyChainRuleCtx(), frule, args...)

function Cassette.execute(::MyChainRuleCtx, ::typeof(frule)
                          ::@domain({R → R}), f, x::Number)
    fx, df = frule(@domain(R → R), f, x)
    if isa(df, Nothing)
        fx, df = (f(x), ẋ -> ẋ .* ForwardDiff.derivative(f, x))
    end
    return fx, df
end
```
=#

frule(::DomainSignature, ::Vararg{Any}) = (nothing, nothing)

rrule(::DomainSignature, ::Vararg{Any}) = (nothing, nothing)

# TODO: Should the default be to whitelist known holomorphic functions, or to
# blacklist known non-holomorphic functions? This implements the latter.
function frule(::@domain({C → C}), f, x)
    fx, df = frule(@domain(R → R), f, x)
    return fx, ẋ -> (df(ẋ), false)
end

function rrule(::@domain({R → R}), f, x)
    fx, df = frule(@domain(R → R), f, x)
    return fx, (x̄, z̄) -> rchain!(x̄, @thunk(df(z̄)))
end

#####
##### `@frule`
#####

"""
    @frule(signature, f(x, y, ...),
                  (df₁_dx, df₁_dy, ...),
                  (df₂_dx, df₂_dy, ...),
                  ...)

Define a method for `frule` using the given domain signature, call
expression, and derivative expressions.

While this macro is convenient for avoiding boilerplate code when implementing
simple forward rules, note that more advanced rules will probably require
overloading `ChainRules.frule` directly. For now, the macro only
supports real-domain rules.

Examples:

`@frule(R → R, sin(x), cos(x))` expands to:

    function frule(::@domain({R → R}), ::typeof(sin), x)
        return sin(x), ẋ -> fchain(ẋ, @thunk(cos(x)))
    end

`@frule(R×R → R, *(x, y), (y, x))` expands to:

    function frule(::@domain({R×R → R}), ::typeof(*), x, y)
        return *(x, y), (ẋ, ẏ) -> fchain(ẋ, @thunk(y), ẏ, @thunk(x))
    end

`@frule(R → R×R, sincos(x), cos(x), -sin(x))` expands to:

    function frule(::@domain({R → R×R}), ::typeof(sincos), x)
        return sincos(x),
               (ẋ -> fchain(ẋ, @thunk(cos(x))),
                ẋ -> fchain(ẋ, @thunk(-sin(x))))
    end

Note that this last case is a good example of a primitive that is more
efficiently implemented with a manual `frule` overload:

    function frule(::@domain({R → R×R}), ::typeof(sincos), x)
        sinx, cosx = sincos(x)
        return (sinx, cosx),
               (ẋ -> fchain(ẋ, @thunk(cosx)),
                ẋ -> fchain(ẋ, @thunk(-sinx)))
    end
"""
macro frule(signature, call, derivs...)
    return generate_rule_definition(signature, call, nothing, derivs...)
end

"""
    @rrule(signature, f(x, y, ...), (f̄₁, f̄₂, ...), df_dx, df_dy, ...)

Define a method for `rrule` using the given domain signature, call
expression, and derivative expressions.

The third argument to the macro is a tuple of symbols naming the adjoints of the
outputs of `f(x, y, ...)` (or just a single symbol if there is only one output).

While this macro is convenient for avoiding boilerplate code when implementing
simple reverse rules, note that more advanced rules will probably require
overloading `ChainRules.rrule` directly. For now, this macro only
supports real-domain rules.

Examples:

`@rrule(R → R, sum(x), ȳ, ȳ)` expands to:

    function rrule(::@domain({R → R}), ::typeof(sum), x)
        return sum(x), (x̄, ȳ) -> rchain!(x̄, @thunk(ȳ))
    end

`@rrule(R×R → R, *(x, y), z̄, z̄ * y', x' * z̄)` expands to:

    function rrule(::@domain({R×R → R}), ::typeof(*), x, y)
        return x * y,
               ((x̄, z̄) -> rchain!(x̄, @thunk(z̄ * y')),
                (ȳ, z̄) -> rchain!(ȳ, @thunk(x' * z̄)))
    end
"""
macro rrule(signature, call, adjoint_names, derivs...)
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
            chain_call = Expr(:call, :fchain)
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
            chain_call = Expr(:call, :rchain!)
            push!(chain_call.args, seed_name)
            push!(chain_call.args, :(@thunk($(esc(derivs[i])))))
            push!(chains, :($(Expr(:tuple, seed_name, adjoint_names...)) -> $chain_call))
        end
    end
    @assert length(chains) > 0
    chains = length(chains) > 1 ? Expr(:tuple, chains...) : chains[1]
    rule_function = isa(adjoint_names, Nothing) ? :frule : :rrule
    if Meta.isexpr(signature, :braces)
        error("domain signature for `@frule`/`@rrule` should NOT be wrapped in {}")
    end
    signature = Expr(:braces, signature)
    return quote
        @assert(@domain($signature) <: DomainSignature{<:Tuple{Vararg{Union{RealDomain, IgnoreDomain}}},
                                                       <:Tuple{Vararg{Union{RealDomain, IgnoreDomain}}}},
                "@frule and @rrule only support real-domain rules right now")
        function $ChainRules.$rule_function(::@domain($signature),
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

# simple `@frule`s

@frule(R → R, abs2(x), x + x)
@frule(R → R, log(x), inv(x))
@frule(R → R, log10(x), inv(x) / log(10f0))
@frule(R → R, log2(x), inv(x) / log(2f0))
@frule(R → R, log1p(x), inv(x + 1))
@frule(R → R, expm1(x), exp(x))
@frule(R → R, sin(x), cos(x))
@frule(R → R, cos(x), -sin(x))
@frule(R → R, sinpi(x), π * cospi(x))
@frule(R → R, cospi(x), -π * sinpi(x))
@frule(R → R, sind(x), (π / 180f0) * cosd(x))
@frule(R → R, cosd(x), -(π / 180f0) * sind(x))
@frule(R → R, asin(x), inv(sqrt(1 - x^2)))
@frule(R → R, acos(x), -inv(sqrt(1 - x^2)))
@frule(R → R, atan(x), inv(1 + x^2))
@frule(R → R, asec(x), inv(abs(x) * sqrt(x^2 - 1)))
@frule(R → R, acsc(x), -inv(abs(x) * sqrt(x^2 - 1)))
@frule(R → R, acot(x), -inv(1 + x^2))
@frule(R → R, asind(x), 180f0 / π / sqrt(1 - x^2))
@frule(R → R, acosd(x), -180f0 / π / sqrt(1 - x^2))
@frule(R → R, atand(x), 180f0 / π / (1 + x^2))
@frule(R → R, asecd(x), 180f0 / π / abs(x) / sqrt(x^2 - 1))
@frule(R → R, acscd(x), -180f0 / π / abs(x) / sqrt(x^2 - 1))
@frule(R → R, acotd(x), -180f0 / π / (1 + x^2))
@frule(R → R, sinh(x), cosh(x))
@frule(R → R, cosh(x), sinh(x))
@frule(R → R, tanh(x), sech(x)^2)
@frule(R → R, coth(x), -(csch(x)^2))
@frule(R → R, asinh(x), inv(sqrt(x^2 + 1)))
@frule(R → R, acosh(x), inv(sqrt(x^2 - 1)))
@frule(R → R, atanh(x), inv(1 - x^2))
@frule(R → R, asech(x), -inv(x * sqrt(1 - x^2)))
@frule(R → R, acsch(x), -inv(abs(x) * sqrt(1 + x^2)))
@frule(R → R, acoth(x), inv(1 - x^2))
@frule(R → R, deg2rad(x), π / 180f0)
@frule(R → R, rad2deg(x), 180f0 / π)

# manually optimized `frule`s

frule(::@domain({R → R}), ::typeof(+), x) = (x, ẋ -> ifelse(ẋ === nothing, false, ẋ))
frule(::@domain({R → R}), ::typeof(-), x) = (x, ẋ -> ifelse(ẋ === nothing, false, -ẋ))
frule(::@domain({R → R}), ::typeof(inv), x) = (u = inv(x); (u, ẋ -> fchain(ẋ, @thunk(-abs2(u)))))
frule(::@domain({R → R}), ::typeof(sqrt), x) = (u = sqrt(x); (u, ẋ -> fchain(ẋ, @thunk(inv(2 * u)))))
frule(::@domain({R → R}), ::typeof(cbrt), x) = (u = cbrt(x); (u, ẋ -> fchain(ẋ, @thunk(inv(3 * u^2)))))
frule(::@domain({R → R}), ::typeof(exp), x) = (u = exp(x); (u, ẋ -> fchain(ẋ, @thunk(u))))
frule(::@domain({R → R}), ::typeof(exp2), x) = (u = exp2(x); (u, ẋ -> fchain(ẋ, @thunk(u * log(2f0)))))
frule(::@domain({R → R}), ::typeof(exp10), x) = (u = exp10(x); (u, ẋ -> fchain(ẋ, @thunk(u * log(10f0)))))
frule(::@domain({R → R}), ::typeof(tan), x) = (u = tan(x); (u, ẋ -> fchain(ẋ, @thunk(1 + u^2))))
frule(::@domain({R → R}), ::typeof(sec), x) = (u = sec(x); (u, ẋ -> fchain(ẋ, @thunk(u * tan(x)))))
frule(::@domain({R → R}), ::typeof(csc), x) = (u = csc(x); (u, ẋ -> fchain(ẋ, @thunk(-u * cot(x)))))
frule(::@domain({R → R}), ::typeof(cot), x) = (u = cot(x); (u, ẋ -> fchain(ẋ, @thunk(-(1 + u^2)))))
frule(::@domain({R → R}), ::typeof(tand), x) = (u = tand(x); (u, ẋ -> fchain(ẋ, @thunk((π / 180f0) * (1 + u^2)))))
frule(::@domain({R → R}), ::typeof(secd), x) = (u = secd(x); (u, ẋ -> fchain(ẋ, @thunk((π / 180f0) * u * tand(x)))))
frule(::@domain({R → R}), ::typeof(cscd), x) = (u = cscd(x); (u, ẋ -> fchain(ẋ, @thunk(-(π / 180f0) * u * cotd(x)))))
frule(::@domain({R → R}), ::typeof(cotd), x) = (u = cotd(x); (u, ẋ -> fchain(ẋ, @thunk(-(π / 180f0) * (1 + u^2)))))
frule(::@domain({R → R}), ::typeof(sech), x) = (u = sech(x); (u, ẋ -> fchain(ẋ, @thunk(-tanh(x) * u))))
frule(::@domain({R → R}), ::typeof(csch), x) = (u = csch(x); (u, ẋ -> fchain(ẋ, @thunk(-coth(x) * u))))

function frule(::@domain({R×R → R}), ::typeof(atan), y, x)
    h = hypot(y, x)
    return atan(y, x), (ẏ, ẋ) -> fchain(ẏ, @thunk(x / h), ẋ, @thunk(y / h))
end

function frule(::@domain({R×R → R}), ::typeof(hypot), x, y)
    h = hypot(x, y)
    return h, (ẋ, ẏ) -> fchain(ẋ, @thunk(x / h), ẏ, @thunk(y / h))
end

function frule(::@domain({R → R×R}), ::typeof(sincos), x)
    sinx, cosx = sincos(x)
    return (sinx, cosx),
           (ẋ -> fchain(ẋ, @thunk(cosx)),
            ẋ -> fchain(ẋ, @thunk(-sinx)))
end

frule(::@domain({C → C}), ::typeof(conj), x) = conj(x), ẋ -> (false, true)

#####
##### reverse rules
#####

@rrule(R → R, sum(x), ȳ, ȳ)
@rrule(R×R → R, +(x, y), z̄, z̄, z̄)
@rrule(R×R → R, *(x, y), z̄, z̄ * y', x' * z̄)

# TODO: This partial derivative extraction should be doable without the extra
# temporaries or preallocation utilized here, but AFAICT such an approach is
# hard to write without relying on inference hacks unless we have something
# akin to https://github.com/JuliaLang/julia/issues/22129
function rrule(::@domain({_×R → R}), ::typeof(broadcast), f, x)
    f_rule = x -> begin
        y, d = frule(@domain(R → R), f, x)
        y, d(one(x))
    end
    applied_f_rule = broadcast(f_rule, x)
    values = map(first, applied_f_rule)
    derivs = map(last, applied_f_rule)
    return values, (x̄, z̄) -> rchain!(x̄, @thunk(broadcasted(*, z̄, derivs)))
end
