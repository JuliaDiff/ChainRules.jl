# Structured matrices


#####
##### `Diagonal`
#####

rrule(::Type{<:Diagonal}, d::AbstractVector) = Diagonal(d), ȳ->(NO_FIELDS, diag(ȳ))

rrule(::typeof(diag), A::AbstractMatrix) = diag(A), ȳ->(NO_FIELDS, Diagonal(ȳ))

#####
##### `Symmetric`
#####

rrule(::Type{<:Symmetric}, A::AbstractMatrix) = Symmetric(A), ȳ->(NO_FIELDS, _symmetric_back(ȳ))

_symmetric_back(ΔΩ) = @thunk(UpperTriangular(ΔΩ) + LowerTriangular(ΔΩ)' - Diagonal(ΔΩ))
_symmetric_back(ΔΩ::Union{Diagonal,UpperTriangular}) = ΔΩ

#####
##### `Adjoint`
#####

# ✖️✖️✖️TODO: Deal with complex-valued arrays as well
rrule(::Type{<:Adjoint}, A::AbstractMatrix{<:Real}) = Adjoint(A), ȳ->(NO_FIELDS, adjoint(ȳ))
rrule(::Type{<:Adjoint}, A::AbstractVector{<:Real}) = Adjoint(A), ȳ->(NO_FIELDS, vec(adjoint(ȳ)))

rrule(::typeof(adjoint), A::AbstractMatrix{<:Real}) = adjoint(A),  ȳ->(NO_FIELDS, adjoint(ȳ))
rrule(::typeof(adjoint), A::AbstractVector{<:Real}) = adjoint(A),  ȳ->(NO_FIELDS, vec(adjoint(ȳ)))

#####
##### `Transpose`
#####

rrule(::Type{<:Transpose}, A::AbstractMatrix) = Transpose(A), ȳ->(NO_FIELDS, transpose(ȳ))
rrule(::Type{<:Transpose}, A::AbstractVector) = Transpose(A), ȳ->(NO_FIELDS, vec(transpose(ȳ)))

rrule(::typeof(transpose), A::AbstractMatrix) = transpose(A), ȳ->(NO_FIELDS, transpose(ȳ))
rrule(::typeof(transpose), A::AbstractVector) = transpose(A), ȳ->(NO_FIELDS, vec(transpose(ȳ)))

#####
##### Triangular matrices
#####

rrule(::Type{<:UpperTriangular}, A::AbstractMatrix) = UpperTriangular(A), ȳ->(NO_FIELDS, Matrix(ȳ))

rrule(::Type{<:LowerTriangular}, A::AbstractMatrix) = LowerTriangular(A), ȳ->(NO_FIELDS, Matrix(ȳ))
