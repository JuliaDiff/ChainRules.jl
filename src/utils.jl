function frule((_, ḟ), ::typeof(ignore), f, args...; kwargs...)
    return f(args...; kwargs...), nothing
end

function rrule(::typeof(ignore), f, args...; kwargs...)
    y = f(args...; kwargs...)
    function ignore_pullback(ȳ)
        return (NO_FIELDS, nothing)
    end
    return y, ignore_pullback
end
