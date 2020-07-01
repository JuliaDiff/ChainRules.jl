function rrule(::typeof(MersenneTwister), args...)
    function MersenneTwister_rrule(ΔΩ)
        return (NO_FIELDS, map(_ -> Zero(), args)...)
    end
    return MersenneTwister(args...), MersenneTwister_rrule
end
