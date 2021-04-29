# Some utility functions for optimizing linear algebra operations that aren't specific
# to any particular rule definition

# F .* (X - X'), overwrites X if possible
function _mulsubtrans!!(X::AbstractMatrix{T}, F::AbstractMatrix{T}) where T<:Real
    k = size(X, 1)
    @inbounds for j = 1:k, i = 1:j  # Iterate the upper triangle
        if i == j
            X[i,i] = zero(T)
        else
            X[i,j], X[j,i] = F[i,j] * (X[i,j] - X[j,i]), F[j,i] * (X[j,i] - X[i,j])
        end
    end
    return X
end
_mulsubtrans!!(X::AbstractZero, F::AbstractZero) = X
_mulsubtrans!!(X::AbstractZero, F::AbstractMatrix{<:Real}) = X
_mulsubtrans!!(X::AbstractMatrix{<:Real}, F::AbstractZero) = F

# I - X, overwrites X
function _eyesubx!(X::AbstractMatrix)
    n, m = size(X)
    @inbounds for j = 1:m, i = 1:n
        X[i,j] = (i == j) - X[i,j]
    end
    return X
end

_extract_imag(x) = complex(0, imag(x))

"""
    _unionall_wrapper(T::Type) -> UnionAll

Return the most general `UnionAll` type union associated with the concrete type `T`.

# Example
```julia
julia> _unionall_wrapper(typeof(Diagonal(1:3)))
Diagonal

julia> _unionall_wrapper(typeof(Symmetric(randn(3, 3))))
Symmetric
````
"""
_unionall_wrapper(::Type{T}) where {T} = T.name.wrapper

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
