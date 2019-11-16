# Structured matrices


#####
##### `Diagonal`
#####

function rrule(::Type{<:Diagonal}, d::AbstractVector)
    function Diagonal_pullback(ȳ)
        return (NO_FIELDS, @thunk(diag(ȳ)))
    end
    return Diagonal(d), Diagonal_pullback
end

function rrule(::typeof(diag), A::AbstractMatrix)
    function diag_pullback(ȳ)
        return (NO_FIELDS, @thunk(Diagonal(ȳ)))
    end
    return diag(A), diag_pullback
end

function rrule(::typeof(*), D::Diagonal{<:Real}, V::AbstractVector{<:Real})
    function times_pullback(Ȳ)
        return (NO_FIELDS, @thunk(Diagonal(Ȳ .* V)), @thunk(D * Ȳ))
    end
    return D * V, times_pullback
end

#####
##### `Symmetric`
#####

function rrule(::Type{<:Symmetric}, A::AbstractMatrix)
    function Symmetric_pullback(ȳ)
        return (NO_FIELDS, @thunk(_symmetric_back(ȳ)))
    end
    return Symmetric(A), Symmetric_pullback
end

_symmetric_back(ΔΩ) = UpperTriangular(ΔΩ) + LowerTriangular(ΔΩ)' - Diagonal(ΔΩ)
_symmetric_back(ΔΩ::Union{Diagonal,UpperTriangular}) = ΔΩ

#####
##### `Adjoint`
#####

# ✖️✖️✖️TODO: Deal with complex-valued arrays as well
function rrule(::Type{<:Adjoint}, A::AbstractMatrix{<:Real})
    function Adjoint_pullback(ȳ)
        return (NO_FIELDS, @thunk(adjoint(ȳ)))
    end
    return Adjoint(A), Adjoint_pullback
end

function rrule(::Type{<:Adjoint}, A::AbstractVector{<:Real})
    function Adjoint_pullback(ȳ)
        return (NO_FIELDS, @thunk(vec(adjoint(ȳ))))
    end
    return Adjoint(A), Adjoint_pullback
end

function rrule(::typeof(adjoint), A::AbstractMatrix{<:Real})
    function adjoint_pullback(ȳ)
        return (NO_FIELDS, @thunk(adjoint(ȳ)))
    end
    return adjoint(A), adjoint_pullback
end

function rrule(::typeof(adjoint), A::AbstractVector{<:Real})
    function adjoint_pullback(ȳ)
        return (NO_FIELDS, @thunk(vec(adjoint(ȳ))))
    end
    return adjoint(A), adjoint_pullback
end

#####
##### `Transpose`
#####

function rrule(::Type{<:Transpose}, A::AbstractMatrix)
    function Transpose_pullback(ȳ)
        return (NO_FIELDS, @thunk transpose(ȳ))
    end
    return Transpose(A), Transpose_pullback
end

function rrule(::Type{<:Transpose}, A::AbstractVector)
    function Transpose_pullback(ȳ)
        return (NO_FIELDS, @thunk vec(transpose(ȳ)))
    end
    return Transpose(A), Transpose_pullback
end

function rrule(::typeof(transpose), A::AbstractMatrix)
    function transpose_pullback(ȳ)
        return (NO_FIELDS, @thunk transpose(ȳ))
    end
    return transpose(A), transpose_pullback
end

function rrule(::typeof(transpose), A::AbstractVector)
    function transpose_pullback(ȳ)
        return (NO_FIELDS, @thunk vec(transpose(ȳ)))
    end
    return transpose(A), transpose_pullback
end

#####
##### Triangular matrices
#####

function rrule(::Type{<:UpperTriangular}, A::AbstractMatrix)
    function UpperTriangular_pullback(ȳ)
        return (NO_FIELDS, @thunk Matrix(ȳ))
    end
    return UpperTriangular(A), UpperTriangular_pullback
end

function rrule(::Type{<:LowerTriangular}, A::AbstractMatrix)
    function LowerTriangular_pullback(ȳ)
        return (NO_FIELDS, @thunk Matrix(ȳ))
    end
    return LowerTriangular(A), LowerTriangular_pullback
end
