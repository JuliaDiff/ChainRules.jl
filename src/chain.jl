# TODO: turn various comments in this file into docstrings

#####
##### `AbstractChainable`
#####
#=
This file defines a custom algebra for chain rule evaluation that factors
complex support, bundle support, zero-elision, etc. into nicely separated
parts.

The main two operations of this algebra are:

`add`: linearly combine partial derivatives (i.e. the `+` part of the
       multivariate chain rule)

`mul`: multiply partial derivatives by a perturbation/sensitivity coefficient
       (i.e. the `*` part of the multivariate chain rule)

Valid arguments to these operations are `T` where `T<:AbstractChainable`, or
where `T` has `broadcast`, `+`, and `*` implementations.

A bunch of the operations in this file have kinda monad-y "fallthrough"
implementations; each step handles an element of the algebra before dispatching
to the next step. This way, we don't need to implement extra machinery just to
resolve ambiguities (e.g. a promotion mechanism).
=#

abstract type AbstractChainable end

# TODO: pick better ordering when possible for matmuls
@inline mul(a, b) = mul_zero(a, b)
@inline mul(a, b, c) = mul(a, mul(b, c))
@inline mul(a, b, c, d) = mul(a, mul(b, c, d))
@inline mul(a, b, c, d, e) = mul(a, mul(b, c, d, e))
@inline mul(a, b, c, d, e, args...) = mul(a, mul(b, c, d, e), args...)

@inline mul_zero(a, b) = mul_dne(a, b)
@inline mul_dne(a, b) = mul_one(a, b)
@inline mul_one(a, b) = mul_thunk(a, b)
@inline mul_thunk(a, b) = mul_wirtinger(a, b)
@inline mul_wirtinger(a, b) = mul_casted(a, b)
@inline mul_casted(a, b) = mul_fallback(a, b)
@inline mul_fallback(a, b) = materialize(a) * materialize(b)

@inline add(a, b) = add_zero(a, b)
@inline add(a, b, c) = add(a, add(b, c))
@inline add(a, b, c, d) = add(a, add(b, c, d))
@inline add(a, b, c, d, e) = add(a, add(b, c, d, e))
@inline add(a, b, c, d, e, args...) = add(a, add(b, c, d, e), args...)

@inline add_zero(a, b) = add_dne(a, b)
@inline add_dne(a, b) = add_one(a, b)
@inline add_one(a, b) = add_thunk(a, b)
@inline add_thunk(a, b) = add_wirtinger(a, b)
@inline add_wirtinger(a, b) = add_casted(a, b)
@inline add_casted(a, b) = add_fallback(a, b)
@inline add_fallback(a, b) = broadcasted(+, a, b)

_adjoint(x) = adjoint(x)

unwrap(x) = x

#####
##### `@chain`
#####

#=
Here are some examples using `@chain` to implement forward- and reverse-mode
chain rules for an intermediary function of the form:

    y₁, y₂ = f(x₁, x₂)

Forward-Mode:

    @chain(∂y₁_∂x₁, ∂y₁_∂x₂)
    @chain(∂y₂_∂x₁, ∂y₂_∂x₂)

    # expands to:
    (ẏ₁, ẋ₁, ẋ₂) -> add(ẏ₁, mul(@thunk(∂y₁_∂x₁), ẋ₁), mul(@thunk(∂y₁_∂x₂), ẋ₂))
    (ẏ₂, ẋ₁, ẋ₂) -> add(ẏ₂, mul(@thunk(∂y₂_∂x₁), ẋ₁), mul(@thunk(∂y₂_∂x₂), ẋ₂))

Reverse-Mode:

    @chain(adjoint(∂y₁_∂x₁), adjoint(∂y₂_∂x₁))
    @chain(adjoint(∂y₁_∂x₂), adjoint(∂y₂_∂x₂))

    # expands to:
    (x̄₁, ȳ₁, ȳ₂) -> add(x̄₁, mul(@thunk(adjoint(∂y₁_∂x₁)), ȳ₁), mul(@thunk(adjoint(∂y₂_∂x₁)), ȳ₂))
    (x̄₂, ȳ₁, ȳ₂) -> add(x̄₂, mul(@thunk(adjoint(∂y₁_∂x₂)), ȳ₁), mul(@thunk(adjoint(∂y₂_∂x₂)), ȳ₂))

Some notation used here:
- `Δᵢ`: a seed (perturbation/sensitivity), i.e. the result of a chain rule evaluation
- `∂ᵢ`: a partial derivative to be multiplied by `Δᵢ` as part of chain rule evaluation
=#
macro chain(∂s...)
    Δs = [Symbol(string(:Δ, i)) for i in 1:length(∂s)]
    ∂Δs = [:(mul(@thunk($(esc(∂s[i]))), $(Δs[i]))) for i in 1:length(∂s)]
    return :((Δ₀, $(Δs...)) -> add(Δ₀, $(∂Δs...)))
end

#####
##### `Thunk`
#####

struct Thunk{F} <: AbstractChainable
    f::F
end

macro thunk(body)
    return :(Thunk(() -> $(esc(body))))
end

@inline (t::Thunk{F})() where {F} = (t.f)()

struct Memoize{F,R} <: AbstractChainable
    thunk::Thunk{F}
    ret::Ref{R}
end

function Memoize(thunk::Thunk)
    R = Core.Compiler.return_type(thunk, ()) # XXX danger zone!
    return Memoize(thunk, Ref{R}())
end

macro memoize(body)
    return :(Memoize(@thunk($(esc(body)))))
end

function (m::Memoize{F,R})()::R where {F, R}
    if !isassigned(m.ret)
        m.ret[] = m.thunk()
    end
    return m.ret[]::R
end

Base.adjoint(x::Union{Memoize,Thunk}) = @thunk(_adjoint(x()))

Base.Broadcast.materialize(x::Union{Thunk,Memoize}) = materialize(x())

mul_thunk(a::Union{Thunk,Memoize}, b::Union{Thunk,Memoize}) = mul(a(), b())
mul_thunk(a::Union{Thunk,Memoize}, b) = mul(a(), b)
mul_thunk(a, b::Union{Thunk,Memoize}) = mul(a, b())

add_thunk(a::Union{Thunk,Memoize}, b::Union{Thunk,Memoize}) = add(a(), b())
add_thunk(a::Union{Thunk,Memoize}, b) = add(a(), b)
add_thunk(a, b::Union{Thunk,Memoize}) = add(a, b())

unwrap(x::Union{Thunk,Memoize}) = x()

#####
##### `Zero`/`DNE`
#####

struct Zero <: AbstractChainable end

Base.adjoint(::Zero) = Zero()

Base.Broadcast.materialize(::Zero) = false

Base.Broadcast.broadcastable(::Zero) = Ref(Zero())

mul_zero(::Zero, ::Zero) = Zero()
mul_zero(::Zero, ::Any) = Zero()
mul_zero(::Any, ::Zero) = Zero()

add_zero(::Zero, ::Zero) = Zero()
add_zero(::Zero, b) = unwrap(b)
add_zero(a, ::Zero) = unwrap(a)

#=
`DNE` is equivalent to `Zero` for the purposes of propagation (i.e. partial
derivatives which don't exist simply do not contribute to a rule's total
derivative), but is maintained as a separate type so that users can check
against it if necessary.

TODO: Should we rethink this? In general, it might not be possible for
users to detect a difference, since `DNE` materializes to `materialize(Zero())`.
It seems like a derivative's `DNE`-ness should be exposed to users in a way
that's just unrelated to the chain rule algebra. Conversely, we want to minimize
the amount of special-casing needed for users writing higher-level rule
definitions/fallbacks, or else things will get unwieldy...maybe we should
make `DNE` a callable "chain" that always basically returns itself?
=#

struct DNE <: AbstractChainable end

Base.adjoint(::DNE) = DNE()

Base.Broadcast.materialize(::DNE) = materialize(Zero())

Base.Broadcast.broadcastable(::DNE) = Ref(DNE())

mul_dne(::DNE, ::DNE) = DNE()
mul_dne(::DNE, ::Any) = DNE()
mul_dne(::Any, ::DNE) = DNE()

add_dne(::DNE, ::DNE) = DNE()
add_dne(::DNE, b) = unwrap(b)
add_dne(a, ::DNE) = unwrap(a)

#####
##### `One`
#####

struct One <: AbstractChainable end

Base.adjoint(::One) = One()

Base.Broadcast.materialize(::One) = true

Base.Broadcast.broadcastable(::One) = Ref(One())

mul_one(::One, ::One) = One()
mul_one(::One, b) = unwrap(b)
mul_one(a, ::One) = unwrap(a)

add_one(a::One, b::One) = add(materialize(a), materialize(b))
add_one(a::One, b) = add(materialize(a), b)
add_one(a, b::One) = add(a, materialize(b))

#####
##### `Wirtinger`
#####

struct Wirtinger{P,C} <: AbstractChainable
    primal::P
    conjugate::C
end

# TODO: check this against conjugation rule in notes
Base.adjoint(w::Wirtinger) = Wirtinger(_adjoint(w.primal), _adjoint(w.conjugate))

function Base.Broadcast.materialize(w::Wirtinger)
    return Wirtinger(materialize(w.primal), materialize(w.conjugate))
end

# TODO: document derivation that leads to this rule (see notes)
function _mul_wirtinger(a::Wirtinger, b::Wirtinger)
    new_primal = add(mul(a.primal, b.primal), mul(a.conjugate, adjoint(b.conjugate)))
    new_conjugate = add(mul(a.primal, b.conjugate), mul(a.conjugate, adjoint(b.primal)))
    return Wirtinger(new_primal, new_conjugate)
end

mul_wirtinger(a::Wirtinger, b) = Wirtinger(mul(a.primal, b), mul(a.conjugate, b))
mul_wirtinger(a, b::Wirtinger) = Wirtinger(mul(a, b.primal), mul(a, b.conjugate))

function add_wirtinger(a::Wirtinger, b::Wirtinger)
    return Wirtinger(add(a.primal, b.primal), add(a.conjugate, b.conjugate))
end

add_wirtinger(a::Wirtinger, b) = Wirtinger(add(a.primal, b), a.conjugate)
add_wirtinger(a, b::Wirtinger) = Wirtinger(add(a, b.primal), b.conjugate)

#####
##### `Casted`/`Broadcasted`
#####
#=
Why define `Casted` at all - why not just use `Broadcasted`?

Basically, we need a mechanism for picking whether the final `mul(a, b)`
fallback is `broadcast(*, a, b)` vs. `materialize(a) * materialize(b)`. It is
easily possible that the caller desires the latter, even if `a` and/or `b`
are `Broadcasted` objects (for example, a series of chain rule evaluations
where the last operation is a mat-mul, but the preceding operation produced
a `Broadcasted` derivative representation). Thus, we can't just pun on
`Broadcasted` to make the fallback decision; we need a type specifically
for this purpose.
=#

struct Casted{V} <: AbstractChainable
    value::V
end

cast(x) = Casted(x)
cast(f, args...) = Casted(broadcasted(f, args...))

_adjoint(c::Broadcasted) = cast(adjoint, c)

Base.adjoint(c::Casted) = cast(adjoint, c.value)

Base.Broadcast.materialize(c::Casted) = materialize(c.value)

mul_eager(a, b) = materialize(mul(a, b))

mul_casted(a::Casted, b::Casted) = broadcasted(mul_eager, a.value, b.value)
mul_casted(a::Casted, b) = broadcasted(mul_eager, a.value, b)
mul_casted(a, b::Casted) = broadcasted(mul_eager, a, b.value)

add_eager(a, b) = materialize(add(a, b))

add_casted(a::Casted, b::Casted) = broadcasted(add_eager, a.value, b.value)
add_casted(a::Casted, b) = broadcasted(add_eager, a.value, b)
add_casted(a, b::Casted) = broadcasted(add_eager, a, b.value)

unwrap(c::Casted) = c.value

#####
##### `MaterializeInto`
#####

struct MaterializeInto{S} <: AbstractChainable
    storage::S
    increment::Bool
    function MaterializeInto(storage, increment::Bool = true)
        return new{typeof(storage)}(storage, increment)
    end
end

function add(a::MaterializeInto, b)
    _materialize!(a.storage, a.increment ? add(a.storage, b) : b)
    return a
end

function _materialize!(a::Wirtinger, b::Wirtinger)
    materialize!(a.primal, b.primal)
    materialize!(a.conjugate, b.conjugate)
    return a
end

function _materialize!(a::Wirtinger, b)
    materialize!(a.primal, unwrap(b))
    materialize!(a.conjugate, Zero())
    return a
end

function _materialize!(a, b::Wirtinger)
    return error("cannot `materialize!` `Wirtinger` into non-`Wirtinger`")
end

_materialize!(a, b) = materialize!(a, unwrap(b))
