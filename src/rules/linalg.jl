#####
##### `sum`
#####

frule(::typeof(sum), x) = (sum(x), Chain(Δx -> sum(Δx)))

rrule(::typeof(sum), x) = (sum(x), Chain(ΔΩ -> cast(ΔΩ)))

#####
##### `dot`
#####

function frule(::typeof(dot), x, y)
    return dot(x, y), Chain((Δx, Δy) -> sum(Δx * cast(y)) + sum(cast(x) * Δy))
end

function rrule(::typeof(dot), x, y)
    return dot(x, y), (Chain(ΔΩ -> ΔΩ * cast(y)), Chain(ΔΩ -> cast(x) * ΔΩ))
end

#####
##### `inv`
#####

function frule(::typeof(inv), x::AbstractArray)
    Ω = inv(x)
    m = @thunk(-Ω)
    return Ω, Chain(Δx -> m * Δx * Ω)
end

function rrule(::typeof(inv), x::AbstractArray)
    Ω = inv(x)
    m = @thunk(-Ω)
    return Ω, Chain(ΔΩ -> m' * ΔΩ * Ω')
end

#####
##### `det`
#####

function frule(::typeof(det), x)
    Ω, m = det(x), @thunk(inv(x))
    return Ω, Chain(Δx -> Ω * tr(extern(m * Δx)))
end

function rrule(::typeof(det), x)
    Ω, m = det(x), @thunk(inv(x)')
    return Ω, Chain(ΔΩ -> Ω * ΔΩ * m)
end

#####
##### `logdet`
#####

function frule(::typeof(logdet), x)
    Ω, m = logdet(x), @thunk(inv(x))
    return Ω, Chain(Δx -> tr(extern(m * Δx)))
end

function rrule(::typeof(logdet), x)
    Ω, m = logdet(x), @thunk(inv(x)')
    return Ω, Chain(ΔΩ -> ΔΩ * m)
end

#####
##### `trace`
#####

frule(::typeof(tr), x) = (tr(x), Chain(Δx -> tr(extern(Δx))))

rrule(::typeof(tr), x) = (tr(x), Chain(ΔΩ -> Diagonal(fill(ΔΩ, size(x, 1)))))
