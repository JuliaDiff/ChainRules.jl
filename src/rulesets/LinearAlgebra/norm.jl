#####
##### `norm`
#####

function frule((_, Δx), ::typeof(norm), x)
    return if isempty(x)
        z = zero(eltype(x))
        az = float(norm(z))
        az, zero(muladd(az, z, az))
    else
        _norm2_forward(x, Δx)
    end
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

function rrule(
    ::typeof(norm),
    x::Union{StridedArray,LinearAlgebra.AbstractTriangular,Diagonal},
    p::Real,
)
    y = LinearAlgebra.norm(x, p)
    function norm_pullback(Δy)
        ∂x = Thunk() do
            return if isempty(x)
                zero.(x) .* (zero(y) * zero(real(Δy)))
            elseif p == 2
                _norm2_back(x, y, Δy)
            elseif p == 1
                _norm1_back(x, y, Δy)
            elseif p == Inf
                _normInf_back(x, y, Δy)
            elseif p == 0
                zero.(x) .* (zero(y) * zero(real(Δy)))
            elseif p == -Inf
                _normInf_back(x, y, Δy)
            else
                _normp_back_x(x, p, y, Δy)
            end
        end
        ∂p = @thunk _normp_back_p(x, p, y, Δy)
        return (NO_FIELDS, ∂x, ∂p)
    end
    norm_pullback(::Zero) = (NO_FIELDS, Zero(), Zero())
    return y, norm_pullback
end
function rrule(
    ::typeof(norm),
    x::Union{StridedArray,LinearAlgebra.AbstractTriangular,Diagonal},
)
    y = LinearAlgebra.norm(x)
    function norm_pullback(Δy)
        ∂x = if isempty(x)
            zero.(x) .* (zero(y) * zero(real(Δy)))
        else
            _norm2_back(x, y, Δy)
        end
        return (NO_FIELDS, ∂x)
    end
    norm_pullback(::Zero) = (NO_FIELDS, Zero())
    return y, norm_pullback
end
function rrule(
    ::typeof(norm),
    x::Union{LinearAlgebra.TransposeAbsVec,LinearAlgebra.AdjointAbsVec},
    p::Real,
)
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

function rrule(
    ::typeof(LinearAlgebra.normp),
    x::Union{StridedArray,LinearAlgebra.AbstractTriangular,Diagonal},
    p,
)
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
    Δu = real(Δy)
    ∂x = broadcast(x) do xi
        r = xi / y
        a = abs(r)
        ∂xi = r * (a^(p - 2) * Δu)
        return ifelse(isfinite(∂xi), ∂xi, zero(∂xi))
    end
    return ∂x
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

function rrule(
    ::typeof(LinearAlgebra.normMinusInf),
    x::Union{StridedArray,LinearAlgebra.AbstractTriangular,Diagonal},
)
    y = LinearAlgebra.normMinusInf(x)
    normMinusInf_pullback(Δy) = (NO_FIELDS, _normInf_back(x, y, Δy))
    normMinusInf_pullback(::Zero) = (NO_FIELDS, Zero())
    return y, normMinusInf_pullback
end

function rrule(
    ::typeof(LinearAlgebra.normInf),
    x::Union{StridedArray,LinearAlgebra.AbstractTriangular,Diagonal},
)
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

function rrule(
    ::typeof(LinearAlgebra.norm1),
    x::Union{StridedArray,LinearAlgebra.AbstractTriangular,Diagonal},
)
    y = LinearAlgebra.norm1(x)
    norm1_pullback(Δy) = (NO_FIELDS, _norm1_back(x, y, Δy))
    norm1_pullback(::Zero) = (NO_FIELDS, Zero())
    return y, norm1_pullback
end

_norm1_back(x, y, Δy) = sign.(x) .* real(Δy)

#####
##### `norm2`
#####

frule((_, Δx), ::typeof(LinearAlgebra.norm2), x) = _norm2_forward(x, Δx)

function rrule(
    ::typeof(LinearAlgebra.norm2),
    x::Union{StridedArray,LinearAlgebra.AbstractTriangular,Diagonal},
)
    y = LinearAlgebra.norm2(x)
    norm2_pullback(Δy) = (NO_FIELDS, _norm2_back(x, y, Δy))
    norm2_pullback(::Zero) = (NO_FIELDS, Zero())
    return y, norm2_pullback
end

function _norm2_forward(x, Δx, Δp = Zero())
    y = LinearAlgebra.norm2(x)
    # since dot product is efficient for pushforward, we don't accumulate in parallel
    ∂y = real(dot(x, Δx)) * pinv(y)
    return y, ∂y
end
_norm2_back(x, y, Δy) = x .* (real(Δy) * pinv(y))

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
    return y, normalize_pullback
end
