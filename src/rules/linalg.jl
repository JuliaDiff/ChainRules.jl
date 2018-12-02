#####
##### `@rule`s
#####

@rule(LinearAlgebra.dot(x, y), (cast(y), cast(x)))

#####
##### custom rules
#####

function frule(::typeof(inv), x::AbstractArray)
    Ω, m = inv(x), @memoize(-Ω)
    return Ω, (Ω̇, ẋ) -> add(Ω̇, mul(m, ẋ, Ω))
end

function rrule(::typeof(inv), x::AbstractArray)
    Ω, m = inv(x), @memoize(-Ω)
    return Ω, (x̄, Ω̄) -> add(x̄, mul(m', Ω̄, Ω'))
end

function frule(::typeof(LinearAlgebra.det), x)
    Ω, m = det(x), @memoize(inv(x))
    return Ω, (Ω̇, ẋ) -> add(Ω̇, mul(Ω, tr(materialize(mul(m, ẋ)))))
end

function rrule(::typeof(LinearAlgebra.det), x)
    Ω, m = det(x), @memoize(inv(x)')
    return Ω, (x̄, Ω̄) -> add(x̄, mul(Ω, Ω̄, m))
end
