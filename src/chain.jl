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

@inline mul(a, b) = mul_zero(a, b)

@inline mul_zero(a, b) = mul_one(a, b)
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

@inline add_zero(a, b) = add_one(a, b)
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
=#
macro chain(∂s...)
    δs = [Symbol(string(:δ, i)) for i in 1:length(∂s)]
    Δs = Any[]
    for i in 1:length(∂s)
        ∂ = esc(∂s[i])
        push!(Δs, :(mul(@thunk($∂), $(δs[i]))))
    end
    return :((δ₀, $(δs...)) -> add(δ₀, $(Δs...)))
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

(t::Thunk{F})() where {F} = (t.f)()

Base.adjoint(t::Thunk) = @thunk(_adjoint(t()))

Base.Broadcast.materialize(t::Thunk) = materialize(t())

mul_thunk(a::Thunk, b::Thunk) = mul(a(), b())
mul_thunk(a::Thunk, b) = mul(a(), b)
mul_thunk(a, b::Thunk) = mul(a, b())

add_thunk(a::Thunk, b::Thunk) = add(a(), b())
add_thunk(a::Thunk, b) = add(a(), b)
add_thunk(a, b::Thunk) = add(a, b())

unwrap(t::Thunk) = t()

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
Equivalent to `Zero` for the purposes of propagation (i.e. partial
derivatives which don't exist simply do not contribute to a rule's total
derivative).
=#

const DNE = Zero

#=
TODO: How should we really handle the above? This is correct w.r.t. propagator
algebra; even if an actual new type `DNE <: AbstractChainable` was defined,
all the rules would be the same. Furthermore, users wouldn't be able to detect
many differences, since `DNE` must materialize to `materialize(Zero())`. Thus,
it seems like a derivative's `DNE`-ness should be exposed to users in a way
that's just unrelated to the chain rule algebra. Conversely, we want to
minimize the amount of special-casing needed for users writing higher-level
rule definitions/fallbacks, or else things will get unwieldy...
=#

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

struct Wirtinger{P, C} <: AbstractChainable
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

struct Casted{V} <: AbstractChainable
    value::V
end

casted(x) = Casted(x)
casted(f, args...) = Casted(broadcasted(f, args...))

_adjoint(c::Broadcasted) = casted(adjoint, c)

Base.adjoint(c::Casted) = casted(adjoint, c.value)

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
