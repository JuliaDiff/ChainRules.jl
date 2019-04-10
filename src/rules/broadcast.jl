#=
TODO: This partial derivative extraction should be doable without the extra
temporaries utilized here, but AFAICT such an approach is hard to write
without relying on inference hacks unless we have something akin to
https://github.com/JuliaLang/julia/issues/22129.
=#
function _cast_diff(f, x)
    element_rule = u -> begin
        fu, du = frule(f, u)
        fu, extern(du(One()))
    end
    results = broadcast(element_rule, x)
    return first.(results), last.(results)
end

function frule(::typeof(broadcast), f, x)
    Ω, ∂x = _cast_diff(f, x)
    return Ω, Rule((_, Δx) -> Δx * cast(∂x))
end

function rrule(::typeof(broadcast), f, x)
    values, derivs = _cast_diff(f, x)
    return values, (DNERule(), Rule(ΔΩ -> ΔΩ * cast(derivs)))
end
