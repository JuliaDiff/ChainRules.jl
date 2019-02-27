#####
##### `sum`
#####

frule(::typeof(sum), x) = (sum(x), Chain((ΔΩ, Δx) -> ΔΩ + sum(Δx)))

rrule(::typeof(sum), x) = (sum(x), Chain((Δx, ΔΩ) -> Δx + cast(ΔΩ)))

#####
##### `dot`
#####

function frule(::typeof(dot), x, y)
    return dot(x, y), Chain((ΔΩ, Δx, Δy) -> ΔΩ + sum(Δx * cast(y)) + sum(cast(x) * Δy))
end

function rrule(::typeof(dot), x, y)
    return dot(x, y), (Chain((Δx, ΔΩ) -> Δx + ΔΩ * cast(y)),
                       Chain((Δy, ΔΩ) -> Δy + cast(x) * ΔΩ))
end

#####
##### `inv`
#####

function frule(::typeof(inv), x::AbstractArray)
    Ω = inv(x)
    m = @thunk(-Ω)
    return Ω, Chain((ΔΩ, Δx) -> ΔΩ + m * Δx * Ω)
end

function rrule(::typeof(inv), x::AbstractArray)
    Ω = inv(x)
    m = @thunk(-Ω)
    return Ω, Chain((Δx, ΔΩ) -> Δx + m' * ΔΩ * Ω')
end

#####
##### `det`
#####

function frule(::typeof(det), x)
    Ω, m = det(x), @thunk(inv(x))
    return Ω, Chain((ΔΩ, Δx) -> ΔΩ + Ω * tr(extern(m * Δx)))
end

function rrule(::typeof(det), x)
    Ω, m = det(x), @thunk(inv(x)')
    return Ω, Chain((Δx, ΔΩ) -> Δx + Ω * ΔΩ * m)
end

#####
##### `logdet`
#####

function frule(::typeof(logdet), x)
    Ω, m = logdet(x), @thunk(inv(x))
    return Ω, Chain((ΔΩ, Δx) -> ΔΩ + tr(extern(m * Δx)))
end

function rrule(::typeof(logdet), x)
    Ω, m = logdet(x), @thunk(inv(x)')
    return Ω, Chain((Δx, ΔΩ) -> Δx + ΔΩ * m)
end

#####
##### `trace`
#####

frule(::typeof(tr), x) = (tr(x), Chain((ΔΩ, Δx) -> ΔΩ + tr(extern(Δx))))

rrule(::typeof(tr), x) = (tr(x), Chain((Δx, ΔΩ) -> Δx + Diagonal(fill(ΔΩ, size(x, 1)))))
