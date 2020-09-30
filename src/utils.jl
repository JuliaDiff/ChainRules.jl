function frule((_, ḟ), ::typeof(ignore), f)
    return f(), nothing
end

function rrule(::typeof(ignore), f)
    y = f()
    function ignore_pullback(ȳ)
        return (NO_FIELDS, nothing)
    end
    return y, ignore_pullback
end