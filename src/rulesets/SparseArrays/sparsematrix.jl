function rrule(::typeof(sparse), I::AbstractVector, J::AbstractVector, V::AbstractVector, m, n, combine::typeof(+))
    project_V = ProjectTo(V)
    
    function sparse_pullback(Ω̄)
        ΔΩ = unthunk(Ω̄)
        ΔV = project_V(ΔΩ[I .+ m .* (J .- 1)])
        return NoTangent(), NoTangent(), NoTangent(), ΔV, NoTangent(), NoTangent(), NoTangent()
    end

    return sparse(I, J, V, m, n, combine), sparse_pullback
end

function rrule(::Type{T}, A::AbstractMatrix) where T <: AbstractSparseMatrix
    function sparse_pullback(Ω̄)
        return NoTangent(), Ω̄
    end
    return T(A), sparse_pullback
end

function rrule(::Type{T}, v::AbstractVector) where T <: AbstractSparseVector
    function sparse_pullback(Ω̄)
        return NoTangent(), Ω̄
    end
    return T(v), sparse_pullback
end

function rrule(::typeof(findnz), A::AbstractSparseMatrix)
    I, J, V = findnz(A)
    m, n = size(A)

    function findnz_pullback(Δ)
        _, _, V̄ = unthunk(Δ)
        V̄ isa AbstractZero && return (NoTangent(), V̄)
        return NoTangent(), sparse(I, J, V̄, m, n)
    end

    return (I, J, V), findnz_pullback
end

function rrule(::typeof(findnz), v::AbstractSparseVector)
    I, V = findnz(v)
    n = length(v)

    function findnz_pullback(Δ)
        _, V̄ = unthunk(Δ)
        V̄ isa AbstractZero && return (NoTangent(), V̄)
        return NoTangent(), sparse(I, V̄, n)
    end

    return (I, V), findnz_pullback
end
