#####
##### `@rule`s
#####

@rule(dot(x, y), (cast(y), cast(x)))
@rule(sum(x), One())

#####
##### custom rules
#####

# inv

function frule(::typeof(inv), x::AbstractArray)
    Ω = inv(x)
    m = @memoize(-Ω)
    return Ω, (Ω̇, ẋ) -> add(Ω̇, mul(m, ẋ, Ω))
end

function rrule(::typeof(inv), x::AbstractArray)
    Ω = inv(x)
    m = @memoize(-Ω)
    return Ω, (x̄, Ω̄) -> add(x̄, mul(m', Ω̄, Ω'))
end

# det

function frule(::typeof(det), x)
    Ω, m = det(x), @memoize(inv(x))
    return Ω, (Ω̇, ẋ) -> add(Ω̇, mul(Ω, tr(materialize(mul(m, ẋ)))))
end

function rrule(::typeof(det), x)
    Ω, m = det(x), @memoize(inv(x)')
    return Ω, (x̄, Ω̄) -> add(x̄, mul(Ω, Ω̄, m))
end

# logdet

function frule(::typeof(LinearAlgebra.logdet), x)
    Ω, m = logdet(x), @memoize(inv(x))
    return Ω, (Ω̇, ẋ) -> add(Ω̇, tr(materialize(mul(m, ẋ))))
end

function rrule(::typeof(LinearAlgebra.logdet), x)
    Ω, m = logdet(x), @memoize(inv(x)')
    return Ω, (x̄, Ω̄) -> add(x̄, mul(Ω̄, m))
end

# trace

frule(::typeof(tr), x) = (tr(x), (Ω̇, ẋ) -> add(Ω̇, Diagonal(ẋ)))

rrule(::typeof(tr), x) = (tr(x), (x̄, Ω̄) -> add(x̄, Diagonal(Ω̄)))
