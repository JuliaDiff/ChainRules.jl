rrule(::Type{<:Diagonal}, d::AbstractVector) = Diagonal(d), Rule(diag)
rrule(::typeof(diag), A::AbstractMatrix) = diag(A), Rule(Diagonal)
