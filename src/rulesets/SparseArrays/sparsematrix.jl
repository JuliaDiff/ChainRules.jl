using SparseArrays

function rrule(::Type{<:SparseMatrixCSC{T,N}}, arr) where {T,N}
    function SparseMatrix_pullback(Δ)
        return NO_FIELDS, collect(Δ)
    end
    return SparseMatrixCSC{T,N}(arr), SparseMatrix_pullback
end

function rrule(::typeof(Matrix), x::SparseMatrixCSC)
  function Matrix_pullback(Δ)
      NO_FIELDS, Δ
  end
  return Matrix(x), Matrix_pullback
end
