# Structured matrices

#####
##### `Diagonal`
#####

rrule(::Type{<:Diagonal}, d::AbstractVector) = Diagonal(d), (NO_FIELDS_RULE, Rule(diag))

rrule(::typeof(diag), A::AbstractMatrix) = diag(A), (NO_FIELDS_RULE, Rule(Diagonal))

#####
##### `Symmetric`
#####

rrule(::Type{<:Symmetric}, A::AbstractMatrix) = Symmetric(A), (NO_FIELDS_RULE, Rule(_symmetric_back))

_symmetric_back(ΔΩ) = UpperTriangular(ΔΩ) + LowerTriangular(ΔΩ)' - Diagonal(ΔΩ)
_symmetric_back(ΔΩ::Union{Diagonal,UpperTriangular}) = ΔΩ

#####
##### `Adjoint`
#####

# TODO: Deal with complex-valued arrays as well
rrule(::Type{<:Adjoint}, A::AbstractMatrix{<:Real}) = Adjoint(A), (NO_FIELDS_RULE, Rule(adjoint))
rrule(::Type{<:Adjoint}, A::AbstractVector{<:Real}) = Adjoint(A), (NO_FIELDS_RULE, Rule(vec∘adjoint))

rrule(::typeof(adjoint), A::AbstractMatrix{<:Real}) = adjoint(A), (NO_FIELDS_RULE, Rule(adjoint))
rrule(::typeof(adjoint), A::AbstractVector{<:Real}) = adjoint(A), (NO_FIELDS_RULE, Rule(vec∘adjoint))

#####
##### `Transpose`
#####

rrule(::Type{<:Transpose}, A::AbstractMatrix) = Transpose(A), (NO_FIELDS_RULE, Rule(transpose))
rrule(::Type{<:Transpose}, A::AbstractVector) = Transpose(A), (NO_FIELDS_RULE, Rule(vec∘transpose))

rrule(::typeof(transpose), A::AbstractMatrix) = transpose(A), (NO_FIELDS_RULE, Rule(transpose))
rrule(::typeof(transpose), A::AbstractVector) = transpose(A), (NO_FIELDS_RULE, Rule(vec∘transpose))

#####
##### Triangular matrices
#####

rrule(::Type{<:UpperTriangular}, A::AbstractMatrix) = UpperTriangular(A), (NO_FIELDS_RULE, Rule(Matrix))

rrule(::Type{<:LowerTriangular}, A::AbstractMatrix) = LowerTriangular(A), (NO_FIELDS_RULE, Rule(Matrix))
