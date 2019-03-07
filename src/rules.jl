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
        fx, df = (f(x), Chain(Δx -> ForwardDiff.derivative(f, x) * Δx))
    else
        fx, df = r
    end
    return fx, df
end
```
=#

frule(::Any, ::Vararg{Any}) = nothing

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
        return Ω, (Chain((Δx₁, Δx₂, ...) -> ∂f₁_∂x₁ * Δx₁ + ∂f₁_∂x₂ * Δx₂ + ...),
                   Chain((Δx₁, Δx₂, ...) -> ∂f₂_∂x₁ * Δx₁ + ∂f₂_∂x₂ * Δx₂ + ...),
                   ...)
    end

    function ChainRules.rrule(::typeof(f), x₁, x₂, ...)
        Ω = f(x₁, x₂, ...)
        \$(statement₁, statement₂, ...)
        return Ω, (Chain((ΔΩ₁, ΔΩ₂, ...) -> ∂f₁_∂x₁ * ΔΩ₁ + ∂f₂_∂x₁ * ΔΩ₂ + ...),
                   Chain((ΔΩ₁, ΔΩ₂, ...) -> ∂f₁_∂x₂ * ΔΩ₁ + ∂f₂_∂x₂ * ΔΩ₂ + ...),
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
        forward_chains = Any[chain_from_partials(partial.args...) for partial in partials]
        reverse_chains = Any[]
        for i in 1:length(inputs)
            reverse_partials = [partial.args[i] for partial in partials]
            push!(reverse_chains, chain_from_partials(reverse_partials...))
        end
    else
        @assert length(inputs) == 1 && all(!Meta.isexpr(partial, :tuple) for partial in partials)
        forward_chains = Any[chain_from_partials(partial) for partial in partials]
        reverse_chains = Any[chain_from_partials(partials...)]
    end
    forward_chains = length(forward_chains) == 1 ? forward_chains[1] : Expr(:tuple, forward_chains...)
    reverse_chains = length(reverse_chains) == 1 ? reverse_chains[1] : Expr(:tuple, reverse_chains...)
    return quote
        function ChainRules.frule(::typeof($f), $(inputs...))
            $(esc(:Ω)) = $call
            $(setup_stmts...)
            return $(esc(:Ω)), $forward_chains
        end
        function ChainRules.rrule(::typeof($f), $(inputs...))
            $(esc(:Ω)) = $call
            $(setup_stmts...)
            return $(esc(:Ω)), $reverse_chains
        end
    end
end

function chain_from_partials(∂s...)
    wirtinger_indices = findall(x -> Meta.isexpr(x, :call) && x.args[1] === :Wirtinger,  ∂s)
    ∂s = map(esc, ∂s)
    Δs = [Symbol(string(:Δ, i)) for i in 1:length(∂s)]
    Δs_tuple = Expr(:tuple, Δs...)
    if isempty(wirtinger_indices)
        ∂_mul_Δs = [:(mul(@thunk($(∂s[i])), $(Δs[i]))) for i in 1:length(∂s)]
        return :(Chain($Δs_tuple -> add($(∂_mul_Δs...))))
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
        primal_chain = :(Chain($Δs_tuple -> add($(∂_mul_Δs_primal...))))
        conjugate_chain = :(Chain($Δs_tuple -> add($(∂_mul_Δs_conjugate...))))
        return quote
            $(∂_wirtinger_defs...)
            WirtingerChain($primal_chain, $conjugate_chain)
        end
    end
end
