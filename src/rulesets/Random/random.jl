frule(Δargs, ::typeof(MersenneTwister), args...) = MersenneTwister(args...), Zero()

function rrule(::typeof(MersenneTwister), args...)
    function MersenneTwister_pullback(ΔΩ)
        return (NO_FIELDS, map(_ -> Zero(), args)...)
    end
    return MersenneTwister(args...), MersenneTwister_pullback
end
