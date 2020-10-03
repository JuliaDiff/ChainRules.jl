
"""
    ignore() do
        ...
    end

Tell ChainRules to ignore a block of code. Everything inside the `do` block will run
on the forward pass as normal, but ChainRules won't try to differentiate it at all.
This can be useful for e.g. code that does logging of the forward pass.

Obviously, you run the risk of incorrect gradients if you use this incorrectly.
"""
ignore(f) = f()

"""
    @ignore (...)

Tell ChainRules to ignore an expression. Equivalent to `ignore() do (...) end`.
Example:

```julia-repl
julia> f(x) = (y = Zygote.@ignore x; x * y);
julia> f'(1)
1
```
"""

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

macro ignore(ex)
    return :(ChainRules.ignore() do 
        $(esc(ex))        
    end)
end
