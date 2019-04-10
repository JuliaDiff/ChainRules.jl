"""
Subtypes of `AbstractRule` are types which represent the primitive derivative
propagation "rules" that can be composed to implement forward- and reverse-mode
automatic differentiation.

More specifically, a `rule::AbstractRule` is a callable Julia object generally
obtained via calling [`frule`](@ref) or [`rrule`](@ref). Such rules accept
differential values as input, evaluate the chain rule using internally stored/
computed partial derivatives to produce a single differential value, then
return that calculated differential value.

For example:

```julia-repl
julia> using ChainRules: frule, rrule, AbstractRule

julia> x, y = rand(2);

julia> h, dh = frule(hypot, x, y);

julia> h == hypot(x, y)
true

julia> isa(dh, AbstractRule)
true

julia> Δx, Δy = rand(2);

julia> dh(Δx, Δy) == ((y / h) * Δx + (x / h) * Δy)
true

julia> h, (dx, dy) = rrule(hypot, x, y);

julia> h == hypot(x, y)
true

julia> isa(dx, AbstractRule) && isa(dy, AbstractRule)
true

julia> Δh = rand();

julia> dx(Δh) == (y / h) * Δh
true

julia> dy(Δh) == (x / h) * Δh
true
```

See also: [`frule`](@ref), [`rrule`](@ref), [`Rule`](@ref), [`DNERule`](@ref), [`WirtingerRule`](@ref)
"""
abstract type AbstractRule end

# this ensures that consumers don't have to special-case rule destructuring
Base.iterate(rule::AbstractRule) = (rule, nothing)
Base.iterate(::AbstractRule, ::Any) = nothing

"""
TODO
"""
accumulate(Δ, rule::AbstractRule, args...) = add(Δ, rule(args...))

"""
TODO
"""
accumulate!(Δ, rule::AbstractRule, args...) = materialize!(Δ, broadcastable(add(cast(Δ), rule(args...))))

"""
TODO
"""
store!(Δ, rule::AbstractRule, args...) = materialize!(Δ, broadcastable(rule(args...)))

#####
##### `Rule`
#####

Cassette.@context RuleContext

const RULE_CONTEXT = Cassette.disablehooks(RuleContext())

Cassette.overdub(::RuleContext, ::typeof(+), a, b) = add(a, b)
Cassette.overdub(::RuleContext, ::typeof(*), a, b) = mul(a, b)

Cassette.overdub(::RuleContext, ::typeof(add), a, b) = add(a, b)
Cassette.overdub(::RuleContext, ::typeof(mul), a, b) = mul(a, b)

"""
    Rule(propation_function)

Return a `Rule` that wraps the given `propation_function`. It is assumed that
`propation_function` is a callable object whose arguments are differential
values, and whose output is a single differential value calculated by applying
internally stored/computed partial derivatives to the input differential
values.

For example:

```
frule(::typeof(*), x, y) = x * y, Rule((Δx, Δy) -> Δx * y + x * Δy)

rrule(::typeof(*), x, y) = x * y, (Rule(ΔΩ -> ΔΩ * y'), Rule(ΔΩ -> x' * ΔΩ))
```

See also: [`frule`](@ref), [`rrule`](@ref), [`accumulate`](@ref), [`accumulate!`](@ref), [`store!`](@ref)
"""
struct Rule{F} <: AbstractRule
    f::F
end

(rule::Rule{F})(args...) where {F} = Cassette.overdub(RULE_CONTEXT, rule.f, args...)

#####
##### `DNERule`
#####

"""
TODO
"""
struct DNERule <: AbstractRule end

DNERule(args...) = DNE()

#####
##### `WirtingerRule`
#####

"""
TODO
"""
struct WirtingerRule{P<:AbstractRule,C<:AbstractRule} <: AbstractRule
    primal::P
    conjugate::C
end

function (rule::WirtingerRule)(args...)
    return Wirtinger(rule.primal(args...), rule.conjugate(args...))
end

#####
##### `frule`/`rrule`
#####

#=
In some weird ideal sense, the fallback for e.g. `frule` should actually be "get
the derivative via forward-mode AD". This is necessary to enable mixed-mode
rules, where e.g. `frule` is used within a `rrule` definition. For example,
broadcasted functions may not themselves be forward-mode *primitives*, but are
often forward-mode *differentiable*.

ChainRules, by design, is decoupled from any specific AD implementation. How,
then, do we know which AD to fall back to when there isn't a primitive defined?

Well, if you're a greedy AD implementation, you can just overload `frule` and/or
`rrule` to use your AD directly. However, this won't play nice with other AD
packages doing the same thing, and thus could cause load-order-dependent
problems for downstream users.

It turns out, Cassette solves this problem nicely by allowing AD authors to
overload the fallbacks w.r.t. their own context. Example using ForwardDiff:

```
using ChainRules, ForwardDiff, Cassette

Cassette.@context MyChainRuleCtx

# ForwardDiff, itself, can call `my_frule` instead of
# `frule` to utilize the ForwardDiff-injected ChainRules
# infrastructure
my_frule(args...) = Cassette.overdub(MyChainRuleCtx(), frule, args...)

function Cassette.execute(::MyChainRuleCtx, ::typeof(frule), f, x::Number)
    r = frule(f, x)
    if isa(r, Nothing)
        fx, df = (f(x), Rule(Δx -> ForwardDiff.derivative(f, x) * Δx))
    else
        fx, df = r
    end
    return fx, df
end
```
=#

"""
    frule(f, x...)

Expressing `x` as the tuple `(x₁, x₂, ...)` and the output tuple of `f(x...)`
as `Ω`, return the tuple:

    (Ω, (rule_for_ΔΩ₁::AbstractRule, rule_for_ΔΩ₂::AbstractRule, ...))

where each returned propagation rule `rule_for_ΔΩᵢ` can be invoked as

    rule_for_ΔΩᵢ(Δx₁, Δx₂, ...)

to yield `Ωᵢ`'s corresponding differential `ΔΩᵢ`. To illustrate, if all involved
values are real-valued scalars, this differential can be written as:

    ΔΩᵢ = ∂Ωᵢ_∂x₁ * Δx₁ + ∂Ωᵢ_∂x₂ * Δx₂ + ...

If no method matching `frule(f, xs...)` has been defined, then return `nothing`.

Examples:

unary input, unary output scalar function:

```julia-repl
julia> x = rand();

julia> sinx, dsin = ChainRules.frule(sin, x);

julia> sinx == sin(x)
true

julia> dsin(1) == cos(x)
true
```

unary input, binary output scalar function:

```julia-repl
julia> x = rand();

julia> sincosx, (dsin, dcos) = ChainRules.frule(sincos, x);

julia> sincosx == sincos(x)
true

julia> dsin(1) == cos(x)
true

julia> dcos(1) == -sin(x)
true
```

See also: [`rrule`](@ref), [`AbstractRule`](@ref), [`@scalar_rule`](@ref)
"""
frule(::Any, ::Vararg{Any}) = nothing

"""
    rrule(f, x...)

Expressing `x` as the tuple `(x₁, x₂, ...)` and the output tuple of `f(x...)`
as `Ω`, return the tuple:

    (Ω, (rule_for_Δx₁::AbstractRule, rule_for_Δx₂::AbstractRule, ...))

where each returned propagation rule `rule_for_Δxᵢ` can be invoked as

    rule_for_Δxᵢ(previous_Δxᵢ, ΔΩ₁, ΔΩ₂, ...)

to yield `xᵢ`'s corresponding differential `Δxᵢ`. To illustrate, if all involved
values are real-valued scalars, this differential can be written as:

    Δxᵢ = ∂Ω₁_∂xᵢ * ΔΩ₁ + ∂Ω₂_∂xᵢ * ΔΩ₂ + ...

If no method matching `rrule(f, xs...)` has been defined, then return `nothing`.

Examples:

unary input, unary output scalar function:

```julia-repl
julia> x = rand();

julia> sinx, dx = ChainRules.rrule(sin, x);

julia> sinx == sin(x)
true

julia> dx(1) == cos(x)
true
```

binary input, unary output scalar function:

```julia-repl
julia> x, y = rand(2);

julia> hypotxy, (dx, dy) = ChainRules.rrule(hypot, x, y);

julia> hypotxy == hypot(x, y)
true

julia> dx(1) == (y / hypot(x, y))
true

julia> dy(1) == (x / hypot(x, y))
true
```

See also: [`frule`](@ref), [`AbstractRule`](@ref), [`@scalar_rule`](@ref)
"""
rrule(::Any, ::Vararg{Any}) = nothing

#####
##### macros
#####

"""
    @scalar_rule(f(x₁, x₂, ...),
                 @setup(statement₁, statement₂, ...),
                 (∂f₁_∂x₁, ∂f₁_∂x₂, ...),
                 (∂f₂_∂x₁, ∂f₂_∂x₂, ...),
                 ...)

A convenience macro that generates simple scalar forward or reverse rules using
the provided partial derivatives. Specifically, generates the corresponding
methods for `frule` and `rrule`:

    function ChainRules.frule(::typeof(f), x₁, x₂, ...)
        Ω = f(x₁, x₂, ...)
        \$(statement₁, statement₂, ...)
        return Ω, (Rule((Δx₁, Δx₂, ...) -> ∂f₁_∂x₁ * Δx₁ + ∂f₁_∂x₂ * Δx₂ + ...),
                   Rule((Δx₁, Δx₂, ...) -> ∂f₂_∂x₁ * Δx₁ + ∂f₂_∂x₂ * Δx₂ + ...),
                   ...)
    end

    function ChainRules.rrule(::typeof(f), x₁, x₂, ...)
        Ω = f(x₁, x₂, ...)
        \$(statement₁, statement₂, ...)
        return Ω, (Rule((ΔΩ₁, ΔΩ₂, ...) -> ∂f₁_∂x₁ * ΔΩ₁ + ∂f₂_∂x₁ * ΔΩ₂ + ...),
                   Rule((ΔΩ₁, ΔΩ₂, ...) -> ∂f₁_∂x₂ * ΔΩ₁ + ∂f₂_∂x₂ * ΔΩ₂ + ...),
                   ...)
    end

Note that the result of `f(x₁, x₂, ...)` is automatically bound to `Ω`. This
allows the primal result to be conveniently referenced (as `Ω`) within the
derivative/setup expressions.

Note that the `@setup` argument can be elided if no setup code is need. In other
words:

    @scalar_rule(f(x₁, x₂, ...),
                 (∂f₁_∂x₁, ∂f₁_∂x₂, ...),
                 (∂f₂_∂x₁, ∂f₂_∂x₂, ...),
                 ...)

is equivalent to:

    @scalar_rule(f(x₁, x₂, ...),
                 @setup(nothing),
                 (∂f₁_∂x₁, ∂f₁_∂x₂, ...),
                 (∂f₂_∂x₁, ∂f₂_∂x₂, ...),
                 ...)

For examples, see ChainRules' `rules` directory.

See also: [`frule`](@ref), [`rrule`](@ref), [`AbstractRule`](@ref)
"""
macro scalar_rule(call, maybe_setup, partials...)
    if Meta.isexpr(maybe_setup, :macrocall) && maybe_setup.args[1] == Symbol("@setup")
        setup_stmts = map(esc, maybe_setup.args[3:end])
    else
        setup_stmts = (nothing,)
        partials = (maybe_setup, partials...)
    end
    @assert Meta.isexpr(call, :call)
    f, inputs = esc(call.args[1]), esc.(call.args[2:end])
    if all(Meta.isexpr(partial, :tuple) for partial in partials)
        forward_rules = Any[rule_from_partials(partial.args...) for partial in partials]
        reverse_rules = Any[]
        for i in 1:length(inputs)
            reverse_partials = [partial.args[i] for partial in partials]
            push!(reverse_rules, rule_from_partials(reverse_partials...))
        end
    else
        @assert length(inputs) == 1 && all(!Meta.isexpr(partial, :tuple) for partial in partials)
        forward_rules = Any[rule_from_partials(partial) for partial in partials]
        reverse_rules = Any[rule_from_partials(partials...)]
    end
    forward_rules = length(forward_rules) == 1 ? forward_rules[1] : Expr(:tuple, forward_rules...)
    reverse_rules = length(reverse_rules) == 1 ? reverse_rules[1] : Expr(:tuple, reverse_rules...)
    return quote
        function ChainRules.frule(::typeof($f), $(inputs...))
            $(esc(:Ω)) = $call
            $(setup_stmts...)
            return $(esc(:Ω)), $forward_rules
        end
        function ChainRules.rrule(::typeof($f), $(inputs...))
            $(esc(:Ω)) = $call
            $(setup_stmts...)
            return $(esc(:Ω)), $reverse_rules
        end
    end
end

function rule_from_partials(∂s...)
    wirtinger_indices = findall(x -> Meta.isexpr(x, :call) && x.args[1] === :Wirtinger,  ∂s)
    ∂s = map(esc, ∂s)
    Δs = [Symbol(string(:Δ, i)) for i in 1:length(∂s)]
    Δs_tuple = Expr(:tuple, Δs...)
    if isempty(wirtinger_indices)
        ∂_mul_Δs = [:(mul(@thunk($(∂s[i])), $(Δs[i]))) for i in 1:length(∂s)]
        return :(Rule($Δs_tuple -> add($(∂_mul_Δs...))))
    else
        ∂_mul_Δs_primal = Any[]
        ∂_mul_Δs_conjugate = Any[]
        ∂_wirtinger_defs = Any[]
        for i in 1:length(∂s)
            if i in wirtinger_indices
                Δi = Δs[i]
                ∂i = Symbol(string(:∂, i))
                push!(∂_wirtinger_defs, :($∂i = $(∂s[i])))
                ∂f∂i_mul_Δ = :(mul(wirtinger_primal($∂i), wirtinger_primal($Δi)))
                ∂f∂ī_mul_Δ̄ = :(mul(conj(wirtinger_conjugate($∂i)), wirtinger_conjugate($Δi)))
                ∂f̄∂i_mul_Δ = :(mul(wirtinger_conjugate($∂i), wirtinger_primal($Δi)))
                ∂f̄∂ī_mul_Δ̄ = :(mul(conj(wirtinger_primal($∂i)), wirtinger_conjugate($Δi)))
                push!(∂_mul_Δs_primal, :(add($∂f∂i_mul_Δ, $∂f∂ī_mul_Δ̄)))
                push!(∂_mul_Δs_conjugate, :(add($∂f̄∂i_mul_Δ, $∂f̄∂ī_mul_Δ̄)))
            else
                ∂_mul_Δ = :(mul(@thunk($(∂s[i])), $(Δs[i])))
                push!(∂_mul_Δs_primal, ∂_mul_Δ)
                push!(∂_mul_Δs_conjugate, ∂_mul_Δ)
            end
        end
        primal_rule = :(Rule($Δs_tuple -> add($(∂_mul_Δs_primal...))))
        conjugate_rule = :(Rule($Δs_tuple -> add($(∂_mul_Δs_conjugate...))))
        return quote
            $(∂_wirtinger_defs...)
            WirtingerRule($primal_rule, $conjugate_rule)
        end
    end
end
