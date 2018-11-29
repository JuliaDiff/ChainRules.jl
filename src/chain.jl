# TODO: turn various comments in this file into docstrings

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
to the next step. This way, we don't need to implement any extra machinery to
resolve ambiguities (e.g. a promotion mechanism).
=#

#####
##### `AbstractChainable`
#####

abstract type AbstractChainable end

# `Wirtinger`: The (primal, conjugate) pair specifying a Wirtinger derivative.

struct Wirtinger{P, C} <: AbstractChainable
    primal::P
    conjugate::C
end

# TODO: check this against conjugation rule in notes
Base.adjoint(w::Wirtinger) = Wirtinger(adjoint(w.primal), adjoint(w.conjugate))

Base.Broadcast.materialize(w::Wirtinger) = Wirtinger(materialize(w.primal), materialize(w.conjugate))

# `One`/`Zero`: Special singleton representations of `1`/`0` enabling static optimizations

struct One <: AbstractChainable end

Base.adjoint(x::One) = x

Base.Broadcast.materialize(::One) = true

struct Zero <: AbstractChainable end

Base.adjoint(x::Zero) = x

Base.Broadcast.materialize(::Zero) = false

# `Thunk`: a representation of a delayed computation

struct Thunk{F} <: AbstractChainable
    f::F
end

macro thunk(body)
    return :(Thunk(() -> $(esc(body))))
end

@inline (thunk::Thunk{F})() where {F} = (thunk.f)()

Base.adjoint(t::Thunk) = @thunk(adjoint(t()))

# Yeah, this is pirated, but it allows callers to skip a cumbersome type-check
# that would otherwise be needed to `adjoint` elements of the `mul`/`add` chain
# rule algebra.
Base.adjoint(b::Base.Broadcast.Broadcasted) = @thunk(adjoint(materialize(b)))

Base.Broadcast.materialize(t::Thunk) = materialize(t())

# `Bundle`: A bundle of tangent/adjoint seeds for "vector-mode" propagation

struct Bundle{P} <: AbstractChainable
    partials::P
end

Base.adjoint(b::Bundle) = Bundle(broadcasted(adjoint, b.partials))

Base.Broadcast.materialize(b::Bundle) = Bundle(materialize(b.partials))

unbundle(x::Bundle) = x.partials
unbundle(x) = x

#####
##### `_materialize!`
#####

@inline _materialize!(a, b) = _materialize_bundle!(a, b)

_materialize_bundle!(a::Bundle, b::Bundle) = (_materialize!(a.partials, b.partials); a)
_materialize_bundle!(a::Bundle, b) = (_materialize!(a.partials, b); a)
_materialize_bundle!(a, b::Bundle) = _materialize!(a, b.partials)
@inline _materialize_bundle!(a, b) = _materialize_zero!(a, b)

_materialize_zero!(::Zero, ::Zero) = error("cannot `materialize!` into `Zero`")
_materialize_zero!(::Zero, ::Any) = error("cannot `materialize!` into `Zero`")
_materialize_zero!(a, b::Zero) = _materialize!(a, materialize(b))
@inline _materialize_zero!(a, b) = _materialize_one!(a, b)

_materialize_one!(::One, ::One) = error("cannot `materialize!` into `One`")
_materialize_one!(::One, ::Any) = error("cannot `materialize!` into `One`")
_materialize_one!(a, b::One) = _materialize!(a, materialize(b))
@inline _materialize_one!(a, b) = _materialize_thunk!(a, b)

_materialize_thunk!(::Thunk, ::Thunk) = error("cannot `materialize!` into `Thunk`")
_materialize_thunk!(::Thunk, ::Any) = error("cannot `materialize!` into `Thunk`")
_materialize_thunk!(a, b::Thunk) = _materialize!(a, b())
@inline _materialize_thunk!(a, b) = _materialize_wirtinger!(a, b)

function _materialize_wirtinger!(a::Wirtinger, b::Wirtinger)
    _materialize!(a.primal, b.primal)
    _materialize!(a.conjugate, b.conjugate)
    return a
end

function _materialize_wirtinger!(a::Wirtinger, b)
    _materialize!(a.primal, b)
    _materialize!(a.conjugate, Zero())
    return a
end

_materialize_wirtinger!(a, b::Wirtinger) = error("cannot `materialize!` `Wirtinger` into non-`Wirtinger`")
@inline _materialize_wirtinger!(a, b) = _materialize_fallback!(a, b)

_materialize_fallback!(a, b) = materialize!(a, b)

#####
##### `add`
#####

struct MaterializeInto{S} <: AbstractChainable
    storage::S
    increment::Bool
    function MaterializeInto(storage, increment::Bool = true)
        storage = materialize(storage)
        return new{typeof(storage)}(storage, increment)
    end
end

function add(a::MaterializeInto, b)
    _materialize!(a.storage, a.increment ? add(a.storage, b) : b)
    return a
end

@inline add(a) = a
@inline add(a, b) = _add_bundle(a, b)
@inline add(a, b, c) = add(a, add(b, c))
@inline add(a, b, c, d) = add(a, add(b, c, d))
@inline add(a, b, c, d, e) = add(a, add(b, c, d, e))
@inline add(a, b, c, d, e, rest...) = add(a, add(b, c, d, e), rest...)

_add_eager(a, b) = materialize(add(a, b))
_add_bundle(a::Bundle, b::Bundle) = Bundle(@thunk(broadcasted(_add_eager, materialize(a.partials), materialize(b.partials))))
_add_bundle(a::Bundle, b) = Bundle(@thunk(broadcasted(_add_eager, materialize(a.partials), unbundle(materialize(b)))))
_add_bundle(a, b::Bundle) = Bundle(@thunk(broadcasted(_add_eager, unbundle(materialize(a)), materialize(b.partials))))
@inline _add_bundle(a, b) = _add_zero(a, b)

_add_zero(::Zero, ::Zero) = Zero()
_add_zero(::Zero, b) = b
_add_zero(a, ::Zero) = a
@inline _add_zero(a, b) = _add_one(a, b)

_add_one(a::One, b::One) = add(materialize(a), materialize(b))
_add_one(a::One, b) = add(materialize(a), b)
_add_one(a, b::One) = add(a, materialize(b))
@inline _add_one(a, b) = _add_thunk(a, b)

_add_thunk(a::Thunk, b::Thunk) = @thunk(add(a(), b()))
_add_thunk(a::Thunk, b) = @thunk(add(a(), b))
_add_thunk(a, b::Thunk) = @thunk(add(a, b()))
@inline _add_thunk(a, b) = _add_wirtinger(a, b)

_add_wirtinger(a::Wirtinger, b::Wirtinger) = Wirtinger(add(a.primal, b.primal),
                                                           add(a.conj, b.conj))
_add_wirtinger(a::Wirtinger, b) = Wirtinger(add(a.primal, b), a.conj)
_add_wirtinger(a, b::Wirtinger) = Wirtinger(add(a, b.primal), b.conj)
@inline _add_wirtinger(a, b) = _add_fallback(a, b)

_add_fallback(a, b) = broadcasted(+, a, b)

#####
##### `mul`
#####

macro mul(a, b)
    return :(mul(@thunk($(esc(a))), @thunk($(esc(b)))))
end

@inline mul(a) = a
@inline mul(a, b) = _mul_bundle(a, b)
@inline mul(a, b, c) = mul(mul(a, b), c)
@inline mul(a, b, c, d) = mul(mul(a, b, c), d)
@inline mul(a, b, c, d, e) = mul(mul(a, b, c, d), e)
@inline mul(a, b, c, d, e, rest...) = mul(mul(a, b, c, d, e), rest...)

_mul_eager(a, b) = materialize(mul(a, b))
_mul_bundle(a::Bundle, b::Bundle) = Bundle(@thunk(broadcasted(_mul_eager, materialize(a.partials), materialize(b.partials))))
_mul_bundle(a::Bundle, b) = Bundle(@thunk(broadcasted(_mul_eager, materialize(a.partials), unbundle(materialize(b)))))
_mul_bundle(a, b::Bundle) = Bundle(@thunk(broadcasted(_mul_eager, unbundle(materialize(a)), materialize(b.partials))))
@inline _mul_bundle(a, b) = _mul_zero(a, b)

_mul_zero(::Zero, ::Zero) = Zero()
_mul_zero(::Zero, b) = Zero()
_mul_zero(a, ::Zero) = Zero()
@inline _mul_zero(a, b) = _mul_one(a, b)

_mul_one(::One, ::One) = One()
_mul_one(::One, b) = b
_mul_one(a, ::One) = a
@inline _mul_one(a, b) = _mul_thunk(a, b)

_mul_thunk(a::Thunk, b::Thunk) = @thunk(mul(a(), b()))
_mul_thunk(a::Thunk, b) = @thunk(mul(a(), b))
_mul_thunk(a, b::Thunk) = @thunk(mul(a, b()))
@inline _mul_thunk(a, b) = _mul_wirtinger(a, b)

# TODO: document derivation that leads to this rule (see notes)
function _mul_wirtinger(a::Wirtinger, b::Wirtinger)
    new_primal = add(mul(a.primal, b.primal), mul(a.conjugate, adjoint(b.conjugate)))
    new_conjugate = add(mul(a.primal, b.conjugate), mul(a.conjugate, adjoint(b.primal)))
    return Wirtinger(new_primal, new_conjugate)
end

_mul_wirtinger(a::Wirtinger, b) = Wirtinger(mul(a.primal, b), mul(a.conj, b))
_mul_wirtinger(a, b::Wirtinger) = Wirtinger(mul(a, b.primal), mul(a, b.conj))
@inline _mul_wirtinger(a, b) = _mul_fallback(a, b)

_mul_fallback(a, b) = a * b

#####
##### `chain`
#####

macro chain(derivatives...)
    seeded_thunks = Any[]
    chained = :chained
    args = [Symbol(string(:seed_, i)) for i in 1:length(derivatives)]
    for i in 1:length(derivatives)
        d = esc(derivatives[i])
        push!(seeded_thunks, :(mul(@thunk($d), $(args[i]))))
    end
    return :(($chained, $(args...)) -> add($chained, $(seeded_thunks...)))
end

#=
Here are some examples using `chain` macro above to implement forward- and
reverse-mode chain rules for an intermediary function of the form:

    y₁, y₂ = f(x₁, x₂)

Forward-Mode:

    @chain(∂y₁_∂x₁, ∂y₁_∂x₂)
    @chain(∂y₂_∂x₁, ∂y₂_∂x₂)

    # expands to:
    (ẏ₁, ẋ₁, ẋ₂) -> add(ẏ₁, @mul(∂y₁_∂x₁, ẋ₁), @mul(∂y₁_∂x₂, ẋ₂))
    (ẏ₂, ẋ₁, ẋ₂) -> add(ẏ₂, @mul(∂y₂_∂x₁, ẋ₁), @mul(∂y₂_∂x₂, ẋ₂))

Reverse-Mode:

    @chain(adjoint(∂y₁_∂x₁), adjoint(∂y₂_∂x₁))
    @chain(adjoint(∂y₁_∂x₂), adjoint(∂y₂_∂x₂))

    # expands to:
    (x̄₁, ȳ₁, ȳ₂) -> add(x̄₁, @mul(∂y₁_∂x₁', ȳ₁), @mul(∂y₂_∂x₁', ȳ₂))
    (x̄₂, ȳ₁, ȳ₂) -> add(x̄₂, @mul(∂y₁_∂x₂', ȳ₁), @mul(∂y₂_∂x₂', ȳ₂))
=#
