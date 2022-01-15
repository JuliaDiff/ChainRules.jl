
#####
##### `+`
#####

function frule((_, Δx, ΔI), ::typeof(+), x::AbstractMatrix, I::UniformScaling)
    return x + I, Δx + ΔI
end

function frule((_, ΔI, Δx), ::typeof(+), I::UniformScaling, x::AbstractMatrix)
    return I + x, ΔI + Δx
end

function rrule(::typeof(+), x::AbstractMatrix, I::UniformScaling)
    project_x = ProjectTo(x)
    project_λ = ProjectTo(I.λ)
    y = x + I
    function plus_back(dy)
        dx = unthunk(dy)
        dλ = if I.λ isa Bool
            NoTangent()
        else
            Tangent{typeof(I)}(; λ = project_λ(tr(dx)))
        end
        return (NoTangent(), project_x(dx), dλ)
    end
    return y, plus_back
end

function rrule(::typeof(+), I::UniformScaling, x::AbstractMatrix)
    y, back = rrule(+, x, I)
    function plus_back_2(dy)
        df, dx, dI = back(dy)
        return (df, dI, dx)
    end
    return y, plus_back_2
end

#####
##### `-`
#####

function frule((_, Δx, ΔI), ::typeof(-), x::AbstractMatrix, I::UniformScaling)
    return x - I, Δx - ΔI
end

function frule((_, ΔI, Δx), ::typeof(-), I::UniformScaling, x::AbstractMatrix)
    return I - x, ΔI - Δx
end

function rrule(::typeof(-), x::AbstractMatrix, I::UniformScaling)
    y, back = rrule(+, x, -I)
    function minus_back_1(dy)
        df, dx, dImaybe = back(dy)
        dI = I.λ isa Bool ? NoTangent() : dImaybe  # as -true isa Int
        return (df, dx, -dI)
    end
    return y, minus_back_1
end

function rrule(::typeof(-), I::UniformScaling, x::AbstractMatrix)
    project_x = ProjectTo(x)
    project_λ = ProjectTo(I.λ)
    y = I - x
    function minus_back_2(dy)
        dx = -unthunk(dy)
        dλ = if I.λ isa Bool
            NoTangent()
        else
            Tangent{typeof(I)}(; λ = project_λ(-tr(dx)))
        end
        return (NoTangent(), dλ, project_x(dx))
    end
    return y, minus_back_2
end

#####
##### `Matrix`
#####

function rrule(::Type{T}, I::UniformScaling, (m, n)) where {T<:AbstractMatrix}
    project_λ = ProjectTo(I.λ)
    function Matrix_back_I(dy)
        if I.λ isa Bool
            return (NoTangent(), NoTangent(), NoTangent())
        end
        dλ = if m == n
            project_λ(tr(unthunk(dy)))
        else
            project_λ(sum(diag(unthunk(dy))))
        end
        return (NoTangent(), Tangent{typeof(I)}(; λ = dλ), NoTangent())
    end
    return T(I, m, n), Matrix_back_I
end


