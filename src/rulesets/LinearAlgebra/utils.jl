# Some utility functions for optimizing linear algebra operations that aren't specific
# to any particular rule definition

_extract_imag(x) = complex(0, imag(x))

"""
    WithSomeZeros{T}

This is a union of LinearAlgebra types, all of which are partly structral zeros,
with a simple backing array given by `parent(x)`. All have methods of `_rewrap`
to re-create.

This exists to solve a type instability, as broadcasting for instance
`λ .* Diagonal(rand(3))` gives a dense matrix when `x==Inf`.
But `withsomezeros_rewrap(x, λ .* parent(x))` is type-stable.
"""
WithSomeZeros{T} = Union{
    Diagonal{T},
    UpperTriangular{T},
    UnitUpperTriangular{T},
    # UpperHessenberg{T},  # doesn't exist in Julia 1.0
    LowerTriangular{T},
    UnitLowerTriangular{T},
}
for S in [
    :Diagonal,
    :UpperTriangular,
    :UnitUpperTriangular,
    # :UpperHessenberg,
    :LowerTriangular,
    :UnitLowerTriangular,
]
    @eval withsomezeros_rewrap(::$S, x) = $S(x)
end

# Bidiagonal, Tridiagonal have more complicated storage.
# AdjOrTransUpperOrUnitUpperTriangular would need adjoint(parent(parent()))
