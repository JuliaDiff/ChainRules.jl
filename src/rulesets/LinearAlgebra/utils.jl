# Some utility functions for optimizing linear algebra operations that aren't specific
# to any particular rule definition

# F .* (X - X'), overwrites X
function _mulsubtrans!(X::AbstractMatrix{T}, F::AbstractMatrix{T}) where T<:Real
    k = size(X, 1)
    @inbounds for j = 1:k, i = 1:j  # Iterate the upper triangle
        if i == j
            X[i,i] = zero(T)
        else
            X[i,j], X[j,i] = F[i,j] * (X[i,j] - X[j,i]), F[j,i] * (X[j,i] - X[i,j])
        end
    end
    X
end

# I - X, overwrites X
function _eyesubx!(X::AbstractMatrix)
    n, m = size(X)
    @inbounds for j = 1:m, i = 1:n
        X[i,j] = (i == j) - X[i,j]
    end
    X
end

# X + Y, overwrites X
function _add!(X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) where T<:Real
    @inbounds for i = eachindex(X, Y)
        X[i] += Y[i]
    end
    X
end
