#####
##### `AbstractDifferential`
#####

"""
The subtypes of `AbstractDifferential` define a custom \"algebra\" for chain
rule evaluation that attempts to factor various features like complex derivative
support, broadcast fusion, zero-elision, etc. into nicely separated parts.

All subtypes of `AbstractDifferential` implement the following operations:

`add(a, b)`: linearly combine differential `a` and differential `b`

`mul(a, b)`: multiply the differential `a` by the differential `b`

`extern(x)`: convert `x` into an appropriate non-`AbstractDifferential` type for use with external packages

Valid arguments to these operations are `T` where `T<:AbstractDifferential`, or
where `T` has proper `+` and `*` implementations.

Additionally, all subtypes of `AbstractDifferential` support `Base.iterate` and
`Base.Broadcast.broadcastable(x)`.
"""
abstract type AbstractDifferential end

"""
    extern(x)

Return `x` converted to an appropriate non-`AbstractDifferential` type, for use
with external packages that might not handle `AbstractDifferential` types.

Note that this function may return an alias (not necessarily a copy) to data
wrapped by `x`, such that mutating `extern(x)` might mutate `x` itself.
"""
@inline extern(x) = x

#=
This `AbstractDifferential` algebra has a monad-y "fallthrough" implementation;
each step handles an element of the algebra before dispatching to the next step.
This way, we don't need to implement promotion/conversion rules between subtypes
of `AbstractDifferential` to resolve potential ambiguities.
=#

const PRECEDENCE_LIST = [:accumulated, :wirtinger, :casted,
                         :zero, :dne, :one, :thunk, :fallback]

global defs = Expr(:block)

let previous_add_name = :add, previous_mul_name = :mul
    for name in PRECEDENCE_LIST
        next_add_name = Symbol(string(:add_, name))
        next_mul_name = Symbol(string(:mul_, name))
        push!(defs.args, quote
            @inline $(previous_add_name)(a, b) = $(next_add_name)(a, b)
            @inline $(previous_mul_name)(a, b) = $(next_mul_name)(a, b)
        end)
        previous_add_name = next_add_name
        previous_mul_name = next_mul_name
    end
end

eval(defs)

@inline add_fallback(a, b) = a + b

@inline mul_fallback(a, b) = a * b

@inline function Base.iterate(x::AbstractDifferential)
    externed = extern(x)
    element, state = iterate(externed)
    return element, (externed, state)
end

@inline function Base.iterate(::AbstractDifferential, (externed, state))
    element, new_state = iterate(externed, state)
    return element, (externed, new_state)
end

#####
##### `Accumulated`
#####

struct Accumulated{S} <: AbstractDifferential
    storage::S
    increment::Bool
    function Accumulated(storage, increment::Bool = true)
        return new{typeof(storage)}(storage, increment)
    end
end

extern(x::Accumulated) = x.storage

function add_accumulated(a::Accumulated, b::Accumulated)
    error("`+(a::Accumulated, b::Accumulated)` is undefined, since its return "*
          "value is ambiguous; it is not possible to determine whether `a` or "*
          "`b` should be returned.")
end

function add_accumulated(a::Accumulated, b)
    materialize!(a.storage, broadcastable(a.increment ? add(cast(a.storage), b) : b))
    return a
end

function add_accumulated(a, b::Accumulated)
    materialize!(b.storage, broadcastable(b.increment ? add(a, cast(b.storage)) : a))
    return b
end

mul_accumulated(a::Accumulated, b::Accumulated) = error("multiplication with `Accumulated` is undefined")

mul_accumulated(a::Accumulated, b) = error("multiplication with `Accumulated` is undefined")

mul_accumulated(a, b::Accumulated) = error("multiplication with `Accumulated` is undefined")

#####
##### `Wirtinger`
#####
# TODO: Document the derivations that lead to all of these rules (see notes)

"""
    Wirtinger(primal::Union{Number,AbstractDifferential},
              conjugate::Union{Number,AbstractDifferential})

Return a `Wirtinger` instance with two directly accessible fields:

- `primal`: the value corresponding to `∂f/∂z * dz`
- `conjugate`: the value corresponding to `∂f/∂z̄ * dz̄`

This `Wirtinger` instance, as a whole, represents the complex differential `df`,
defined in the Wirtinger calculus as:

```
df = ∂f/∂z * dz + ∂f/∂z̄ * dz̄
```

This representation allows convenient derivative definitions for nonholomorphic
functions of complex variables. For example, consider the `@scalar_rule` for
`abs2`:

```
@scalar_rule(abs2(x), Wirtinger(x', x))
```
"""
struct Wirtinger{P,C} <: AbstractDifferential
    primal::P
    conjugate::C
    function Wirtinger(primal::Union{Number,AbstractDifferential},
                       conjugate::Union{Number,AbstractDifferential})
        return new{typeof(primal),typeof(conjugate)}(primal, conjugate)
    end
    function Wirtinger(primal, conjugate)
        error("`Wirtinger` only supports elements of type <: Union{Number,AbstractDifferential} for now")
    end
end

"""
    Wirtinger(primal::Real, conjugate::Real)

Return `add(primal, conjugate)`.

The Wirtinger calculus generally requires that downstream propagation mechanisms
have access to `∂f/∂z * dz` and `∂f/∂z̄ * dz` separately. However, if both of
these terms are real-valued, then downstream Wirtinger propagation mechanisms
resolve to the same mechanisms as real-valued calculus. In this case, the sum
in the differential `df = ∂f/∂z * dz + ∂f/∂z̄ * dz` can be computed eagerly and
a special `Wirtinger` representation is not needed.

Thus, this method primarily exists as an optimization.
"""
Wirtinger(primal::Real, conjugate::Real) = add(primal, conjugate)

extern(x::Wirtinger) = error("`Wirtinger` cannot be converted into an external type.")

Base.Broadcast.broadcastable(w::Wirtinger) = Wirtinger(broadcastable(w.primal),
                                                       broadcastable(w.conjugate))

Base.iterate(x::Wirtinger) = (x, nothing)
Base.iterate(::Wirtinger, ::Any) = nothing

function add_wirtinger(a::Wirtinger, b::Wirtinger)
    return Wirtinger(add(a.primal, b.primal), add(a.conjugate, b.conjugate))
end

add_wirtinger(a::Wirtinger, b) = add(a, Wirtinger(b, Zero()))
add_wirtinger(a, b::Wirtinger) = add(Wirtinger(a, Zero()), b)

function mul_wirtinger(a::Wirtinger, b::Wirtinger)
    new_primal = add(mul(a.primal, b.primal), mul(a.conjugate, conj(b.conjugate)))
    new_conjugate = add(mul(a.primal, b.conjugate), mul(a.conjugate, conj(b.primal)))
    return Wirtinger(new_primal, new_conjugate)
end

mul_wirtinger(a::Wirtinger, b) = mul(a, Wirtinger(b, Zero()))
mul_wirtinger(a, b::Wirtinger) = mul(Wirtinger(a, Zero()), b)

#####
##### `Thunk`
#####

struct Thunk{F} <: AbstractDifferential
    f::F
end

macro thunk(body)
    return :(Thunk(() -> $(esc(body))))
end

@inline extern(x::Thunk{F}) where {F} = x.f()

Base.Broadcast.broadcastable(x::Thunk) = broadcastable(extern(x))

add_thunk(a::Thunk, b::Thunk) = add(extern(a), extern(b))
add_thunk(a::Thunk, b) = add(extern(a), b)
add_thunk(a, b::Thunk) = add(a, extern(b))

mul_thunk(a::Thunk, b::Thunk) = mul(extern(a), extern(b))
mul_thunk(a::Thunk, b) = mul(extern(a), b)
mul_thunk(a, b::Thunk) = mul(a, extern(b))

#####
##### `Zero`
#####

struct Zero <: AbstractDifferential end

extern(x::Zero) = false

Base.Broadcast.broadcastable(::Zero) = Ref(Zero())

Base.iterate(x::Zero) = (x, nothing)
Base.iterate(::Zero, ::Any) = nothing

add_zero(::Zero, ::Zero) = Zero()
add_zero(::Zero, b) = b
add_zero(a, ::Zero) = a

mul_zero(::Zero, ::Zero) = Zero()
mul_zero(::Zero, ::Any) = Zero()
mul_zero(::Any, ::Zero) = Zero()

#####
##### `DNE`
#####

struct DNE <: AbstractDifferential end

extern(x::DNE) = error("`DNE` cannot be converted into an external type.")

Base.Broadcast.broadcastable(::DNE) = Ref(DNE())

Base.iterate(x::DNE) = (x, nothing)
Base.iterate(::DNE, ::Any) = nothing

add_dne(::DNE, ::DNE) = DNE()
add_dne(::DNE, b) = b
add_dne(a, ::DNE) = a

mul_dne(::DNE, ::DNE) = DNE()
mul_dne(::DNE, ::Any) = DNE()
mul_dne(::Any, ::DNE) = DNE()

#####
##### `One`
#####

struct One <: AbstractDifferential end

extern(x::One) = true

Base.Broadcast.broadcastable(::One) = Ref(One())

Base.iterate(x::One) = (x, nothing)
Base.iterate(::One, ::Any) = nothing

add_one(a::One, b::One) = add(extern(a), extern(b))
add_one(a::One, b) = add(extern(a), b)
add_one(a, b::One) = add(a, extern(b))

mul_one(::One, ::One) = One()
mul_one(::One, b) = b
mul_one(a, ::One) = a

#####
##### `Casted`
#####

struct Casted{V} <: AbstractDifferential
    value::V
end

cast(x) = Casted(x)
cast(f, args...) = Casted(broadcasted(f, args...))

extern(x::Casted) = materialize(broadcasted(extern, x.value))

Base.Broadcast.broadcastable(x::Casted) = x.value

Base.iterate(x::Casted) = iterate(x.value)
Base.iterate(x::Casted, state) = iterate(x.value, state)

add_casted(a::Casted, b::Casted) = Casted(broadcasted(add, a.value, b.value))
add_casted(a::Casted, b) = Casted(broadcasted(add, a.value, b))
add_casted(a, b::Casted) = Casted(broadcasted(add, a, b.value))

mul_casted(a::Casted, b::Casted) = Casted(broadcasted(mul, a.value, b.value))
mul_casted(a::Casted, b) = Casted(broadcasted(mul, a.value, b))
mul_casted(a, b::Casted) = Casted(broadcasted(mul, a, b.value))
