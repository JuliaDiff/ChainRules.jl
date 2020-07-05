# Some utility functions for optimizing linear algebra operations that aren't specific
# to any particular rule definition

# F .* (X - X'), overwrites X if possible
function _mulsubtrans!(X::AbstractMatrix{T}, F::AbstractMatrix{T}) where T<:Real
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
_mulsubtrans!(X::AbstractZero, F::AbstractZero) = X
_mulsubtrans!(X::AbstractZero, F::AbstractMatrix{<:Real}) = X
_mulsubtrans!(X::AbstractMatrix{<:Real}, F::AbstractZero) = F

# I - X, overwrites X
function _eyesubx!(X::AbstractMatrix)
    n, m = size(X)
    @inbounds for j = 1:m, i = 1:n
        X[i,j] = (i == j) - X[i,j]
    end
    return X
end

# X + Y, overwrites X if possible
function _add!(X::AbstractVecOrMat, Y::AbstractVecOrMat)
    @inbounds for i = eachindex(X, Y)
        X[i] += Y[i]
    end
    return X
end
_add!(X, Y) = X + Y  # handles all `AbstractZero` overloads
