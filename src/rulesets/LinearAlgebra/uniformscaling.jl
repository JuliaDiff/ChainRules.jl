
#####
##### constructor
#####

function rrule(::Type{T}, x::Number) where {T<:UniformScaling}
    UniformScaling_back(dx) = (NoTangent(), ProjectTo(x)(unthunk(dx).λ))
    return T(x), UniformScaling_back
end

#####
##### `+`
#####

function frule((_, Δx, ΔJ), ::typeof(+), x::AbstractMatrix, J::UniformScaling)
    return x + J, Δx + (zero(J) + ΔJ)  # This (0 + ΔJ) allows for ΔJ::Tangent{UniformScaling}
end

function frule((_, ΔJ, Δx), ::typeof(+), J::UniformScaling, x::AbstractMatrix)
    return J + x, (zero(J) + ΔJ) + Δx
end

function rrule(::typeof(+), x::AbstractMatrix, J::UniformScaling)
    project_x = ProjectTo(x)
    project_J = ProjectTo(J)
    function plus_back(dy)
        dx = unthunk(dy)
        return (NoTangent(), project_x(dx), project_J(I * tr(dx)))
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
    return x - J, Δx - (zero(J) + ΔJ)
end

function frule((_, ΔJ, Δx), ::typeof(-), J::UniformScaling, x::AbstractMatrix)
    return J - x, (zero(J) + ΔJ) - Δx
end

function rrule(::typeof(-), x::AbstractMatrix, J::UniformScaling)
    y, back = rrule(+, x, -J)
    project_J = ProjectTo(J)
    function minus_back_1(dy)
        df, dx, dJ = back(dy)
        return (df, dx, project_J(-dJ))  # re-project as -true isa Int
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

function rrule(::Type{T}, I::UniformScaling{<:Bool}, (m, n)) where {T<:AbstractMatrix}
    Matrix_back_I(dy) = (NoTangent(), NoTangent(), NoTangent())
    return T(I, m, n), Matrix_back_I
end

function rrule(::Type{T}, J::UniformScaling, (m, n)) where {T<:AbstractMatrix}
    project_J = ProjectTo(J)
    function Matrix_back_I(dy)
        dJ = if m == n
            project_J(I * tr(unthunk(dy)))
        else
            project_J(I * sum(diag(unthunk(dy))))
        end
        return (NoTangent(), dJ, NoTangent())
    end
    return T(J, m, n), Matrix_back_I
end

