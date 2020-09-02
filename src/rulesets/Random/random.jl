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
@non_differentiable randn()
@non_differentiable randn(::AbstractRNG)
@non_differentiable copy(::Random._GLOBAL_RNG)
@non_differentiable copy(::MersenneTwister)
@non_differentiable copy!(::MersenneTwister, ::MersenneTwister)
@non_differentiable copy!(::MersenneTwister, ::Random._GLOBAL_RNG)
@non_differentiable copy!(::Random._GLOBAL_RNG, ::MersenneTwister)
