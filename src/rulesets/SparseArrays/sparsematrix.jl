using SparseArrays

function rrule(::Type{<:SparseMatrixCSC{T,N}}, arr) where {T,N}
    function SparseMatrix_pullback(Δ)
      return NO_FIELDS, collect(Δ)
    end
    return SparseMatrixCSC{T,N}(arr), SparseMatrix_pullback
end
