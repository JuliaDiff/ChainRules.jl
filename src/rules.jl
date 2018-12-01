#####
##### `frule`/`rrule`
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
                          f, x::Number)
    r = frule(f, x)
    if isa(r, Nothing)
        fx, df = (f(x), @chain(ForwardDiff.derivative(f, x)))
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
    @rule(f(x₁, x₂, ...),
          @setup(statement₁, statement₂, ...),
          (df₁_dx₁, df₁_dx₂, ...),
          (df₂_dx₁, df₂_dx₂, ...),
          ...)

Define the corresponding methods for `frule` and `rrule`:

   function ChainRules.frule(::typeof(f), x₁, x₂, ...)
       Ω = f(x₁, x₂, ...)
       \$(statement₁, statement₂, ...)
       return Ω, (@chain(df₁_dx₁, df₁_dx₂, ...),
                  @chain(df₂_dx₁, df₂_dx₂, ...),
                  ...)
   end

   function ChainRules.rrule(::typeof(f), x₁, x₂, ...)
       Ω = f(x₁, x₂, ...)
       \$(statement₁, statement₂, ...)
       return Ω, (@chain(adjoint(df₁_dx₁), adjoint(df₂_dx₁), ...),
                  @chain(adjoint(df₁_dx₂), adjoint(df₂_dx₂), ...),
                  ...)
   end

Note that the result of `f(x₁, x₂, ...)` is automatically bound to `Ω`. This
allows the primal result to be conveniently referenced (as `Ω`) within the
derivative/setup expressions.

Note that the `@setup` argument can be elided if no setup code is need. In other
words:

    @rule(f(x₁, x₂, ...),
          (df₁_dx₁, df₁_dx₂, ...),
          (df₂_dx₁, df₂_dx₂, ...),
          ...)

is equivalent to:

    @rule(f(x₁, x₂, ...),
          @setup(nothing),
          (df₁_dx₁, df₁_dx₂, ...),
          (df₂_dx₁, df₂_dx₂, ...),
          ...)

While `@rule` is convenient for avoiding boilerplate code for simple forward or
reverse rules, note that more advanced rules will probably require overloading
`ChainRules.frule` or `ChainRules.rrule`  directly.

For examples, see the ChainRules' `rules` directory.
"""
macro rule(call, maybe_setup, partials...)
    if Meta.isexpr(maybe_setup, :macrocall) && maybe_setup.args[1] == Symbol("@setup")
        setup_stmts = map(esc, maybe_setup.args[3:end])
    else
        setup_stmts = (nothing,)
        partials = (maybe_setup, partials...)
    end
    @assert Meta.isexpr(call, :call)
    f, inputs = esc(call.args[1]), esc.(call.args[2:end])
    if all(Meta.isexpr(partial, :tuple) for partial in partials)
        escaped_partials = Any[map(esc, partial.args) for partial in partials]
        forward_chains = Any[:(@chain($(partial...))) for partial in escaped_partials]
        reverse_chains = Any[]
        for i in 1:length(inputs)
            adjoint_partials = [:(_adjoint($(partial[i]))) for partial in escaped_partials]
            push!(reverse_chains, :(@chain($(adjoint_partials...))))
        end
    else
        @assert length(inputs) == 1 && all(!Meta.isexpr(partial, :tuple) for partial in partials)
        escaped_partials = map(esc, partials)
        forward_chains = Any[:(@chain($partial)) for partial in escaped_partials]
        adjoint_partials = Any[:(_adjoint($partial)) for partial in escaped_partials]
        reverse_chains = Any[:(@chain($(adjoint_partials...)))]
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

#####
##### forward-mode --> reverse-mode rule transformation
#####
#=
TODO: expand implementation to arbitrary input/output arity

TODO: implement reverse-mode --> forward-mode rule transformation

TODO: This code isn't actually used anywhere, but could be nice for tests/examples
=#

function reverse_from_forward(f, args...)
    result = frule(f, args...)
    isa(result, Nothing) && return nothing
    f, df_f = result
    df_r = _reverse_from_forward(args, df_f)
    isa(df_r, Nothing) && return nothing
    return f, df_r
end

_reverse_from_forward(::Any, ::Any) = nothing

_reverse_from_forward(::NTuple{1, Any}, df) = (x̄, z̄) -> add(x̄, df(Zero(), z̄')')

function _reverse_from_forward(::NTuple{2, Any}, df)
    return (x̄, z̄) -> add(x̄, df(Zero(), z̄', Zero())'),
           (ȳ, z̄) -> add(ȳ, df(Zero(), Zero(), z̄')')
end

function _reverse_from_forward(::NTuple{1, Any}, df::NTuple{2})
    df₁, df₂ = df
    return (x̄, z̄₁, z̄₂) -> add(x̄, df₁(Zero(), z̄₁')', df₂(Zero(), z̄₂')')
end

function _reverse_from_forward(::NTuple{2, Any}, df::NTuple{2})
    df₁, df₂ = df
    return (x̄, z̄₁, z̄₂) -> add(x̄, df₁(Zero(), z̄₁', Zero())', df₂(Zero(), z̄₂', Zero())')
           (ȳ, z̄₁, z̄₂) -> add(ȳ, df₁(Zero(), Zero(), z̄₁')', df₂(Zero(), Zero(), z̄₂')')
end
