# Structured matrices

#####
##### `Diagonal`
#####

rrule(::Type{<:Diagonal}, d::AbstractVector) = Diagonal(d), Rule(diag)

rrule(::typeof(diag), A::AbstractMatrix) = diag(A), Rule(Diagonal)

#####
##### `Symmetric`
#####

rrule(::Type{<:Symmetric}, A::AbstractMatrix) = Symmetric(A), Rule(_symmetric_back)

_symmetric_back(ΔΩ) = UpperTriangular(ΔΩ) + LowerTriangular(ΔΩ)' - Diagonal(ΔΩ)
_symmetric_back(ΔΩ::Union{Diagonal,UpperTriangular}) = ΔΩ

#####
##### `Adjoint`
#####

# TODO: Deal with complex-valued arrays as well
rrule(::Type{<:Adjoint}, A::AbstractMatrix{<:Real}) = Adjoint(A), Rule(adjoint)
rrule(::Type{<:Adjoint}, A::AbstractVector{<:Real}) = Adjoint(A), Rule(vec∘adjoint)

rrule(::typeof(adjoint), A::AbstractMatrix{<:Real}) = adjoint(A), Rule(adjoint)
rrule(::typeof(adjoint), A::AbstractVector{<:Real}) = adjoint(A), Rule(vec∘adjoint)

#####
##### `Transpose`
#####

rrule(::Type{<:Transpose}, A::AbstractMatrix) = Transpose(A), Rule(transpose)
rrule(::Type{<:Transpose}, A::AbstractVector) = Transpose(A), Rule(vec∘transpose)

rrule(::typeof(transpose), A::AbstractMatrix) = transpose(A), Rule(transpose)
rrule(::typeof(transpose), A::AbstractVector) = transpose(A), Rule(vec∘transpose)

#####
##### Triangular matrices
#####

rrule(::Type{<:UpperTriangular}, A::AbstractMatrix) = UpperTriangular(A), Rule(Matrix)

rrule(::Type{<:LowerTriangular}, A::AbstractMatrix) = LowerTriangular(A), Rule(Matrix)
