frule(Δargs, ::Type{MersenneTwister}, args...) = MersenneTwister(args...), Zero()

function rrule(::Type{MersenneTwister}, args...)
    function MersenneTwister_pullback(ΔΩ)
        return (NO_FIELDS, map(_ -> Zero(), args)...)
    end
    return MersenneTwister(args...), MersenneTwister_pullback
end

@non_differentiable Broadcast.broadcastable(::AbstractRNG)

@non_differentiable Random.randexp(::AbstractRNG)
@non_differentiable Random.randstring(::AbstractRNG)

@non_differentiable rand()
@non_differentiable rand(::AbstractRNG)
@non_differentiable rand(::AbstractRNG, ::Random.Sampler)
@non_differentiable rand(::AbstractRNG, ::Integer)
@non_differentiable rand(::AbstractRNG, ::Integer, ::Integer)
@non_differentiable rand(::AbstractRNG, ::Integer, ::Integer, ::Integer)
@non_differentiable rand(::AbstractRNG, ::Integer, ::Integer, ::Integer, ::Integer)
@non_differentiable rand(::AbstractRNG, ::Integer, ::Integer, ::Integer, ::Integer, ::Integer)
@non_differentiable rand(::Type{<:Real})
@non_differentiable rand(::Type{<:Real}, ::Tuple)
@non_differentiable rand(::Type{<:Real}, ::Integer)
@non_differentiable rand(::Type{<:Real}, ::Integer, ::Integer)
@non_differentiable rand(::Type{<:Real}, ::Integer, ::Integer, ::Integer)
@non_differentiable rand(::Type{<:Real}, ::Integer, ::Integer, ::Integer, ::Integer)
@non_differentiable rand(::Type{<:Real}, ::Integer, ::Integer, ::Integer, ::Integer, ::Integer)
@non_differentiable rand(::Integer)
@non_differentiable rand(::Integer, ::Integer)
@non_differentiable rand(::Integer, ::Integer, ::Integer)
@non_differentiable rand(::Integer, ::Integer, ::Integer, ::Integer)
@non_differentiable rand(::Integer, ::Integer, ::Integer, ::Integer, ::Integer)


# There are many different 1-3 arg methods, but not varargs
@non_differentiable rand!(::Any)
@non_differentiable rand!(::Any, ::Any)
@non_differentiable rand!(::Any, ::Any, ::Any)

@non_differentiable randexp(::Any)
@non_differentiable randexp(::Any, ::Any)
@non_differentiable randexp(::Any, ::Any, ::Any)
@non_differentiable randexp(::Any, ::Any, ::Any, ::Any)
@non_differentiable randexp(::Any, ::Any, ::Any, ::Any, ::Any)

@non_differentiable randexp!(::AbstractArray)
@non_differentiable randexp!(::AbstractRNG, ::AbstractArray)

@non_differentiable randn()
@non_differentiable randn(::Any)
@non_differentiable randn(::Any, ::Any)
@non_differentiable randn(::Any, ::Any, ::Any)
@non_differentiable randn(::Any, ::Any, ::Any, ::Any)
@non_differentiable randn(::Any, ::Any, ::Any, ::Any, ::Any)

@non_differentiable randn!(::AbstractArray)
@non_differentiable randn!(::AbstractRNG, ::AbstractArray)


@non_differentiable randn(::AbstractRNG)
@non_differentiable copy(::AbstractRNG)
@non_differentiable copy!(::AbstractRNG, ::AbstractRNG)

@static if VERSION > v"1.3"
    @non_differentiable Random.default_rng()
    @non_differentiable Random.default_rng(::Int)
end
