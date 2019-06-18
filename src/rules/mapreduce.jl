#####
##### `map`
#####

function rrule(::typeof(map), f, xs...)
    y = map(f, xs...)
    ∂xs = ntuple(length(xs)) do i
        Rule() do ȳ
            map(ȳ, xs...) do ȳi, xis...
                r = rrule(f, xis...)
                if r === nothing
                    throw(ArgumentError("can't differentiate `map` with `$f`; no `rrule` " *
                                        "is defined for `$f$xis`"))
                end
                _, ∂xis = r
                extern(∂xis[i](ȳi))
            end
        end
    end
    return y, (DNERule(), ∂xs...)
end

#####
##### `sum`
#####

frule(::typeof(sum), x) = (sum(x), Rule(sum))

rrule(::typeof(sum), x) = (sum(x), Rule(cast))
