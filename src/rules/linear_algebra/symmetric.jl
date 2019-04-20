rrule(::typeof(Symmetric), A::AbstractMatrix) = Symmetric(A), Rule(_symmetric_back)

_symmetric_back(Δ) = UpperTriangular(Δ) + LowerTriangular(Δ)' - Diagonal(Δ)
_symmetric_back(Δ::Union{Diagonal, UpperTriangular}) = Δ
