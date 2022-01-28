
#####
##### `+`
#####

function frule((_, Δx, ΔJ), ::typeof(+), x::AbstractMatrix, J::UniformScaling)
    return x + J, Δx + ΔJ
end

function frule((_, ΔJ, Δx), ::typeof(+), J::UniformScaling, x::AbstractMatrix)
    return J + x, ΔJ + Δx
end

function rrule(::typeof(+), x::AbstractMatrix, J::UniformScaling)
    project_x = ProjectTo(x)
    project_J = ProjectTo(J)
    function plus_back(dy)
        dx = unthunk(dy)
        (NoTangent(), project_x(dx), project_J(I * tr(dx)))
    end
    return x + J, plus_back
end

function rrule(::typeof(+), J::UniformScaling, x::AbstractMatrix)
    y, back = rrule(+, x, J)
    function plus_back_2(dy)
        df, dx, dJ = back(dy)
        return (df, dJ, dx)
    end
    return y, plus_back_2
end

#####
##### `-`
#####

function frule((_, Δx, ΔJ), ::typeof(-), x::AbstractMatrix, J::UniformScaling)
    return x - J, Δx - ΔJ
end

function frule((_, ΔJ, Δx), ::typeof(-), J::UniformScaling, x::AbstractMatrix)
    return J - x, ΔJ - Δx
end

function rrule(::typeof(-), x::AbstractMatrix, J::UniformScaling)
    y, back = rrule(+, x, -J)
    function minus_back_1(dy)
        df, dx, dJmaybe = back(dy)
        dJ = J.λ isa Bool ? NoTangent() : dJmaybe  # as -true isa Int
        return (df, dx, -dJ)
    end
    return y, minus_back_1
end

function rrule(::typeof(-), J::UniformScaling, x::AbstractMatrix)
    project_x = ProjectTo(x)
    project_J = ProjectTo(J)
    function minus_back_2(dy)
        dx = -unthunk(dy)
        return (NoTangent(), project_J(-tr(dx) * I), project_x(dx))
    end
    return J - x, minus_back_2
end

#####
##### `Matrix`
#####

function rrule(::Type{T}, J::UniformScaling, (m, n)) where {T<:AbstractMatrix}
    project_J = ProjectTo(J)
    function Matrix_back_I(dy)
        if J.λ isa Bool
            return (NoTangent(), NoTangent(), NoTangent())
        end
        dJ = if m == n
            project_J(I * tr(unthunk(dy)))
        else
            project_J(I * sum(diag(unthunk(dy))))
        end
        return (NoTangent(), dJ, NoTangent())
    end
    return T(J, m, n), Matrix_back_I
end

