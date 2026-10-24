frule(Δargs, T::Type{<:AbstractRNG}, args...) = T(args...), ZeroTangent()

function rrule(T::Type{<:AbstractRNG}, args...)
    function AbstractRNG_pullback(ΔΩ)
        return (NoTangent(), map(Returns(ZeroTangent()), args)...)
    end
    return T(args...), AbstractRNG_pullback
end

@non_differentiable Broadcast.broadcastable(::AbstractRNG)

@non_differentiable Random.randstring(::Any...)

# Only samplers whose output does not depend on any differentiable data, or that sample from
# ranges of Base integer types: e.g. not the ones for sampling from arrays, non-integer ranges,
# tuples, or user-defined types.
const NonDifferentiableSampler = Union{
    Random.SamplerType,
    Random.SamplerTrivial{<:Random.FloatInterval},
    Random.SamplerBigFloat,
    Random.SamplerSimple{<:Union{String,SubString{String},AbstractArray{Char}}},
    Random.SamplerRangeInt{<:Base.BitInteger},
    Random.SamplerRangeFast,
    Random.SamplerRangeNDL{<:Any,<:Base.BitInteger},
    Random.SamplerBigInt,
}
@non_differentiable rand(::AbstractRNG, ::NonDifferentiableSampler)
@non_differentiable rand!(::AbstractRNG, ::AbstractArray, ::NonDifferentiableSampler)

@non_differentiable randexp(::Any...)

@non_differentiable randexp!(::AbstractArray)
@non_differentiable randexp!(::AbstractRNG, ::AbstractArray)

@non_differentiable randn(::Any...)

@non_differentiable randn!(::AbstractArray)
@non_differentiable randn!(::AbstractRNG, ::AbstractArray)

@non_differentiable copy(::AbstractRNG)
@non_differentiable copy!(::AbstractRNG, ::AbstractRNG)

@non_differentiable Random.default_rng()
@non_differentiable Random.default_rng(::Int)
