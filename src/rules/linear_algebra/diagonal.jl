rrule(::typeof(Diagonal), d::AbstractVector) = Diagonal(d), Rule(ΔΩ->diag(ΔΩ))
rrule(::typeof(diag), A::AbstractMatrix) = diag(A), Rule(ΔΩ->Diagonal(ΔΩ))
