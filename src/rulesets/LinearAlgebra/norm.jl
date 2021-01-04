#####
##### `norm`
#####

function frule((_, Δx), ::typeof(norm), x)
    y = norm(x)
    return y, _norm2_forward(x, Δx, norm(x))
end
function frule((_, Δx), ::typeof(norm), x::Number, p::Real)
    y = norm(x, p)
    ∂y = if iszero(Δx) || iszero(p)
        zero(real(x)) * zero(real(Δx))
    else
        signx = x isa Real ? sign(x) : x * pinv(y)
        _realconjtimes(signx, Δx)
    end
    return y, ∂y
end

function rrule(::typeof(norm), x::AbstractArray, p::Real)
    y = LinearAlgebra.norm(x, p)
    function norm_pullback_p(Δy)
        ∂x = Thunk() do
            return if isempty(x) || p == 0
                InplaceableThunk(
                    @thunk(zero.(x) .* (zero(y) * zero(real(Δy)))),
                    dx -> dx .= zero(eltype(dx)),
                )
            elseif p == 2
                InplaceableThunk(
                    @thunk(_norm2_back(x, y, Δy)),
                    dx -> _norm2_back!(dx, x, y, Δy),
                )
            elseif p == 1
                InplaceableThunk(
                    @thunk(_norm1_back(x, y, Δy)),
                    dx -> _norm1_back!(dx, x, y, Δy),
                )
            elseif p == Inf
                _normInf_back(x, y, Δy)
            elseif p == -Inf
                _normInf_back(x, y, Δy)
            else
                _normp_back_x(x, p, y, Δy)
            end
        end
        ∂p = @thunk _normp_back_p(x, p, y, Δy)
        return (NO_FIELDS, ∂x, ∂p)
    end
    norm_pullback_p(::Zero) = (NO_FIELDS, Zero(), Zero())
    return y, norm_pullback_p
end
function rrule(::typeof(norm), x::AbstractArray)
    y = LinearAlgebra.norm(x)
    function norm_pullback_2(Δy)
        ∂x = if isempty(x)
            zero.(x) .* (zero(y) * zero(real(Δy)))
        else
            InplaceableThunk(
                @thunk(_norm2_back(x, y, Δy)),
                dx -> _norm2_back!(dx, x, y, Δy),
            )
        end
        return (NO_FIELDS, ∂x)
    end
    norm_pullback_2(::Zero) = (NO_FIELDS, Zero())
    return y, norm_pullback_2
end
function rrule(::typeof(norm), x::Union{LinearAlgebra.AdjOrTransAbsVec}, p::Real)
    y, inner_pullback = rrule(norm, parent(x), p)
    function norm_pullback(Δy)
        (∂self, ∂x′, ∂p) = inner_pullback(Δy)
        fdual = x isa Transpose ? transpose : adjoint
        ∂x = @thunk fdual(unthunk(∂x′))
        return (∂self, ∂x, ∂p)
    end
    return y, norm_pullback
end
function rrule(::typeof(norm), x::Number, p::Real)
    y = norm(x, p)
    function norm_pullback(Δy)
        ∂x = if iszero(Δy) || iszero(p)
            zero(x) * zero(real(Δy))
        else
            signx = x isa Real ? sign(x) : x * pinv(y)
            signx * real(Δy)
        end
        return (NO_FIELDS, ∂x, Zero())
    end
    norm_pullback(::Zero) = (NO_FIELDS, Zero(), Zero())
    return y, norm_pullback
end

#####
##### `normp`
#####

function rrule(::typeof(LinearAlgebra.normp), x::AbstractArray, p)
    y = LinearAlgebra.normp(x, p)
    function normp_pullback(Δy)
        ∂x = @thunk _normp_back_x(x, p, y, Δy)
        ∂p = @thunk _normp_back_p(x, p, y, Δy)
        return (NO_FIELDS, ∂x, ∂p)
    end
    normp_pullback(::Zero) = (NO_FIELDS, Zero(), Zero())
    return y, normp_pullback
end

function _normp_back_x(x, p, y, Δy)
    c = real(Δy) / y
    ∂x = map(x) do xi
        a = norm(xi)
        ∂xi = xi * ((a / y)^(p - 2) * c)
        return ifelse(isfinite(∂xi), ∂xi, zero(∂xi))
    end
    return ∂x
end
function _normp_back_x(x::WithSomeZeros, p, y, Δy) # Diagonal, UpperTriangular, etc.
    c = real(Δy) / y
    ∂x_data = map(parent(x)) do xi
        a = norm(xi)
        ∂xi = xi * ((a / y)^(p - 2) * c)
        return ifelse(isfinite(∂xi), ∂xi, zero(∂xi))
    end
    return withsomezeros_rewrap(x, ∂x_data)
end

function _normp_back_p(x, p, y, Δy)
    y > 0 && isfinite(y) && !iszero(p) || return zero(real(Δy)) * zero(y) / one(p)
    s = sum(x) do xi
        a = norm(xi)
        c = (a / y)^(p - 1) * a * log(a)
        return ifelse(isfinite(c), c, zero(c))
    end
    ∂p = real(Δy) * (s - y * log(y)) / p
    return ∂p
end

#####
##### `normMinusInf`/`normInf`
#####

function rrule(::typeof(LinearAlgebra.normMinusInf), x::AbstractArray)
    y = LinearAlgebra.normMinusInf(x)
    normMinusInf_pullback(Δy) = (NO_FIELDS, _normInf_back(x, y, Δy))
    normMinusInf_pullback(::Zero) = (NO_FIELDS, Zero())
    return y, normMinusInf_pullback
end

function rrule(::typeof(LinearAlgebra.normInf), x::AbstractArray)
    y = LinearAlgebra.normInf(x)
    normInf_pullback(Δy) = (NO_FIELDS, _normInf_back(x, y, Δy))
    normInf_pullback(::Zero) = (NO_FIELDS, Zero())
    return y, normInf_pullback
end

function _normInf_back(x, y, Δy)
    Δu = real(Δy)
    T = typeof(zero(float(eltype(x))) * zero(Δu))
    ∂x = fill!(similar(x, T), 0)
    # if multiple `xi`s have the exact same norm, then they must have been identically
    # produced, e.g. with `fill`. So we set only one to be non-zero.
    # we choose last index to match the `frule`.
    yind = findlast(xi -> norm(xi) == y, x)
    yind === nothing && throw(ArgumentError("y is not the correct norm of x"))
    @inbounds ∂x[yind] = sign(x[yind]) * Δu
    return ∂x
end

#####
##### `norm1`
#####

function rrule(::typeof(LinearAlgebra.norm1), x::AbstractArray)
    y = LinearAlgebra.norm1(x)
    norm1_pullback(Δy) = (NO_FIELDS, InplaceableThunk(
        @thunk(_norm1_back(x, y, Δy)),
        dx -> _norm1_back!(dx, x, y, Δy),
    ))
    norm1_pullback(::Zero) = (NO_FIELDS, Zero())
    return y, norm1_pullback
end

function _norm1_back(x, y, Δy)
    ∂x = sign.(x) .* real(Δy)
    return ∂x
end
function _norm1_back(x::WithSomeZeros, y, Δy)
    ∂x_data = sign.(parent(x)) .* real(Δy)
    return withsomezeros_rewrap(x, ∂x_data)
end
function _norm1_back!(∂x, x, y, Δy)
    ∂x .+= sign.(x) .* real(Δy)
    return ∂x
end

#####
##### `norm2`
#####

function frule((_, Δx), ::typeof(LinearAlgebra.norm2), x)
    y = LinearAlgebra.norm2(x)
    return y, _norm2_forward(x, Δx, y)
end

function rrule(::typeof(LinearAlgebra.norm2), x::AbstractArray)
    y = LinearAlgebra.norm2(x)
    norm2_pullback(Δy) = (NO_FIELDS, InplaceableThunk(
        @thunk(_norm2_back(x, y, Δy)),
        dx -> _norm2_back!(dx, x, y, Δy),
    ))
    norm2_pullback(::Zero) = (NO_FIELDS, Zero())
    return y, norm2_pullback
end

function _norm2_forward(x, Δx, y)
    ∂y = real(dot(x, Δx)) * pinv(y)
    return ∂y
end
function _norm2_back(x, y, Δy)
    ∂x = x .* (real(Δy) * pinv(y))
    return ∂x
end
function _norm2_back(x::WithSomeZeros, y, Δy)
    T = typeof(one(eltype(x)) / one(real(eltype(Δy))))
    ∂x_data = parent(x) .* (real(Δy) * pinv(y))
    return withsomezeros_rewrap(x, ∂x_data)
end
function _norm2_back!(∂x, x, y, Δy)
    ∂x .+= x .* (real(Δy) * pinv(y))
    return ∂x  # must return after mutating
end

#####
##### `normalize`
#####

function rrule(::typeof(normalize), x::AbstractVector, p::Real)
    nrm, inner_pullback = rrule(norm, x, p)
    Ty = typeof(first(x) / nrm)
    y = copyto!(similar(x, Ty), x)
    LinearAlgebra.__normalize!(y, nrm)
    function normalize_pullback(Δy)
        invnrm = pinv(nrm)
        ∂nrm = -dot(y, Δy) * invnrm
        (_, ∂xnorm, ∂p) = inner_pullback(∂nrm)
        ∂x = @thunk unthunk(∂xnorm) .+ Δy .* invnrm
        return (NO_FIELDS, ∂x, ∂p)
    end
    normalize_pullback(::Zero) = (NO_FIELDS, Zero(), Zero())
    return y, normalize_pullback
end
function rrule(::typeof(normalize), x::AbstractVector)
    nrm = LinearAlgebra.norm2(x)
    Ty = typeof(first(x) / nrm)
    y = copyto!(similar(x, Ty), x)
    LinearAlgebra.__normalize!(y, nrm)
    function normalize_pullback(Δy)
        ∂x = (Δy .- real(dot(y, Δy)) .* y) .* pinv(nrm)
        return (NO_FIELDS, ∂x)
    end
    normalize_pullback(::Zero) = (NO_FIELDS, Zero())
    return y, normalize_pullback
end
