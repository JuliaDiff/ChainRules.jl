using SparseArrays

function rrule(<:Type{SparseMatrixCSC{T,N}}, arr) where {T,N}
  function SparseMatrix_pullback(Δ)
    (collect(Δ),)
  end
  SparseMatrixCSC{T,N}(arr), SparseMatrix_pullback
end

function rrlue(::typeof(diagm), x)
  diagm(x), d -> (diag(d),)
end

function rrlue(::typeof(issymmetric), x)
  issymmetric(x), d -> (d,)
end
