rrule(::Type{<:Symmetric}, A::AbstractMatrix) = Symmetric(A), Rule(_symmetric_back)

_symmetric_back(ΔΩ) = UpperTriangular(ΔΩ) + LowerTriangular(ΔΩ)' - Diagonal(ΔΩ)
_symmetric_back(ΔΩ::Union{Diagonal, UpperTriangular}) = ΔΩ
