abstract type AbstractChain end

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

@inline (chain::Chain{F})(args...) where {F} = Cassette.overdub(CHAIN_CONTEXT, chain.f, args...)

# this ensures that consumers don't have to special-case chain destructuring
Base.iterate(chain::Chain) = (chain, nothing)
Base.iterate(::Chain, ::Any) = nothing

#####
##### `ChainDNE`
#####

struct ChainDNE <: AbstractChain end

ChainDNE(args...) = DNE()
