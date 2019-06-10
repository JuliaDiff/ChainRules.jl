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
