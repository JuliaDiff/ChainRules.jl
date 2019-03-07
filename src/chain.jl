abstract type AbstractChain end

# this ensures that consumers don't have to special-case chain destructuring
Base.iterate(chain::AbstractChain) = (chain, nothing)
Base.iterate(::AbstractChain, ::Any) = nothing

#####
##### `Accumulate`
#####

struct Accumulate{S}
    storage::S
    increment::Bool
    function Accumulate(storage, increment::Bool = true)
        return new{typeof(storage)}(storage, increment)
    end
end

function accumulate!(Δ, ∂, increment = true)
    materialize!(Δ, broadcastable(increment ? add(cast(Δ), ∂) : ∂))
    return Δ
end

#####
##### `Chain`
#####

Cassette.@context ChainContext

const CHAIN_CONTEXT = Cassette.disablehooks(ChainContext())

Cassette.overdub(::ChainContext, ::typeof(+), a, b) = add(a, b)
Cassette.overdub(::ChainContext, ::typeof(*), a, b) = mul(a, b)

Cassette.overdub(::ChainContext, ::typeof(add), a, b) = add(a, b)
Cassette.overdub(::ChainContext, ::typeof(mul), a, b) = mul(a, b)

struct Chain{F} <: AbstractChain
    f::F
end

(chain::Chain{F})(Δ, args...) where {F} = add(Δ, Cassette.overdub(CHAIN_CONTEXT, chain.f, args...))

function (chain::Chain{F})(Δ::Accumulate, args...) where {F}
    ∂ = Cassette.overdub(CHAIN_CONTEXT, chain.f, args...)
    return accumulate!(Δ.storage, ∂, Δ.increment)
end

#####
##### `AccumulatorChain`
#####

struct AccumulatorChain{F} <: AbstractChain
    f::F
end

(chain::AccumulatorChain{F})(args...) where {F} = Cassette.overdub(CHAIN_CONTEXT, chain.f, args...)

#####
##### `DNEChain`
#####

struct DNEChain <: AbstractChain end

DNEChain(args...) = DNE()

#####
##### `WirtingerChain`
#####

struct WirtingerChain{P<:AbstractChain,C<:AbstractChain} <: AbstractChain
    primal::P
    conjugate::C
end

function (chain::WirtingerChain)(args...)
    return Wirtinger(chain.primal(args...), chain.conjugate(args...))
end
