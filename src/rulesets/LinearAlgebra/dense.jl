#####
##### `dot`
#####

function frule((_, Δx, Δy), ::typeof(dot), x, y)
    return dot(x, y), dot(Δx, y) + dot(x, Δy)
end

function rrule(::typeof(dot), x, y)
    function dot_pullback(ΔΩ)
        return (NO_FIELDS, @thunk(y .* ΔΩ'), @thunk(x .* ΔΩ))
    end
    return dot(x, y), dot_pullback
end

function frule((_, Δx, ΔA, Δy), ::typeof(dot), x::AbstractVector{<:Number}, A::AbstractMatrix{<:Number}, y::AbstractVector{<:Number})
    return dot(x, A, y), dot(Δx, A, y) + dot(x, ΔA, y) + dot(x, A, Δy)
end

function rrule(::typeof(dot), x::AbstractVector{<:Number}, A::AbstractMatrix{<:Number}, y::AbstractVector{<:Number})
    Ay = A * y
    z = adjoint(x) * Ay
    function dot_pullback(ΔΩ)
        dx = @thunk conj(ΔΩ) .* Ay
        dA = @thunk ΔΩ .* x .* adjoint(y)
        dy = @thunk ΔΩ .* (adjoint(A) * x)
        return (NO_FIELDS, dx, dA, dy)
    end
    dot_pullback(::Zero) = (NO_FIELDS, Zero(), Zero(), Zero())
    return z, dot_pullback
end

function rrule(::typeof(dot), x::AbstractVector{<:Number}, A::Diagonal{<:Number}, y::AbstractVector{<:Number})
    z = dot(x,A,y)
    function dot_pullback(ΔΩ)
        dx = @thunk conj(ΔΩ) .* A.diag .* y  # A*y is this broadcast, can be fused
        dA = @thunk Diagonal(ΔΩ .* x .* conj(y))  # calculate N not N^2 elements
        dy = @thunk ΔΩ .* conj.(A.diag) .* x
        return (NO_FIELDS, dx, dA, dy)
    end
    dot_pullback(::Zero) = (NO_FIELDS, Zero(), Zero(), Zero())
    return z, dot_pullback
end

#####
##### `cross`
#####

function frule((_, Δa, Δb), ::typeof(cross), a::AbstractVector, b::AbstractVector)
    return cross(a, b), cross(Δa, b) .+ cross(a, Δb)
end

# TODO: support complex vectors
function rrule(::typeof(cross), a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    Ω = cross(a, b)
    function cross_pullback(ΔΩ)
        return (NO_FIELDS, @thunk(cross(b, ΔΩ)), @thunk(cross(ΔΩ, a)))
    end
    return Ω, cross_pullback
end


#####
##### `det`
#####

function frule((_, Δx), ::typeof(det), x::AbstractMatrix)
    Ω = det(x)
    # TODO Performance optimization: probably there is an efficent
    # way to compute this trace without during the full compution within
    return Ω, Ω * tr(x \ Δx)
end
frule((_, Δx), ::typeof(det), x::Number) = (det(x), Δx)

function rrule(::typeof(det), x::Union{Number, AbstractMatrix})
    Ω = det(x)
    function det_pullback(ΔΩ)
        ∂x = x isa Number ? ΔΩ : Ω * ΔΩ * inv(x)'
        return (NO_FIELDS, ∂x)
    end
    return Ω, det_pullback
end

#####
##### `logdet`
#####

function frule((_, Δx), ::typeof(logdet), x::Union{Number, StridedMatrix{<:Number}})
    Ω = logdet(x)
    return Ω, tr(x \ Δx)
end

function rrule(::typeof(logdet), x::Union{Number, StridedMatrix{<:Number}})
    Ω = logdet(x)
    function logdet_pullback(ΔΩ)
        ∂x = x isa Number ? ΔΩ / x' : ΔΩ * inv(x)'
        return (NO_FIELDS, ∂x)
    end
    return Ω, logdet_pullback
end

#####
##### `logabsdet`
#####

function frule((_, Δx), ::typeof(logabsdet), x::AbstractMatrix)
    Ω = logabsdet(x)
    (y, signy) = Ω
    b = tr(x \ Δx)
    ∂y = real(b)
    ∂signy = eltype(x) <: Real ? Zero() : im * imag(b) * signy
    ∂Ω = Composite{typeof(Ω)}(∂y, ∂signy)
    return Ω, ∂Ω
end

function rrule(::typeof(logabsdet), x::AbstractMatrix)
    Ω = logabsdet(x)
    function logabsdet_pullback(ΔΩ)
        (Δy, Δsigny) = ΔΩ
        (_, signy) = Ω
        f = signy' * Δsigny
        imagf = f - real(f)
        g = real(Δy) + imagf
        ∂x = g * inv(x)'
        return (NO_FIELDS, ∂x)
    end
    return Ω, logabsdet_pullback
end

#####
##### `trace`
#####

function frule((_, Δx), ::typeof(tr), x)
    return tr(x), tr(Δx)
end

function rrule(::typeof(tr), x)
    # This should really be a FillArray
    # see https://github.com/JuliaDiff/ChainRules.jl/issues/46
    function tr_pullback(ΔΩ)
        return (NO_FIELDS, Diagonal(fill(ΔΩ, size(x, 1))))
    end
    return tr(x), tr_pullback
end


#####
##### `pinv`
#####

@scalar_rule pinv(x) -(Ω ^ 2)

function frule(
    (_, Δx),
    ::typeof(pinv),
    x::AbstractVector{T},
    tol::Real = 0,
) where {T<:Union{Real,Complex}}
    y = pinv(x, tol)
    ∂y′ = sum(abs2, parent(y)) .* Δx .- 2real(y * Δx) .* parent(y)
    ∂y = y isa Transpose ? transpose(∂y′) : adjoint(∂y′)
    return y, ∂y
end

function frule(
    (_, Δx),
    ::typeof(pinv),
    x::LinearAlgebra.AdjOrTransAbsVec{T},
    tol::Real = 0,
) where {T<:Union{Real,Complex}}
    y = pinv(x, tol)
    ∂y = sum(abs2, y) .* vec(Δx') .- 2real(Δx * y) .* y
    return y, ∂y
end

# Formula for derivative adapted from Eq 4.12 of
# Golub, Gene H., and Victor Pereyra. "The Differentiation of Pseudo-Inverses and Nonlinear
# Least Squares Problems Whose Variables Separate."
# SIAM Journal on Numerical Analysis 10(2). (1973). 413-432. doi: 10.1137/0710036
function frule((_, ΔA), ::typeof(pinv), A::AbstractMatrix{T}; kwargs...) where {T}
    Y = pinv(A; kwargs...)
    m, n = size(A)
    # contract over the largest dimension
    if m ≤ n
        ∂Y = -Y * (ΔA * Y)
        ∂Y = add!!(∂Y, (ΔA' - Y * (A * ΔA')) * (Y' * Y))  # (I - Y A) ΔA' Y' Y
        ∂Y = add!!(∂Y, Y * (Y' * ΔA') * (I - A * Y))  # Y Y' ΔA' (I - A Y)
    else
        ∂Y = -(Y * ΔA) * Y
        ∂Y = add!!(∂Y, (I - Y * A) * (ΔA' * Y') * Y)  # (I - Y A) ΔA' Y' Y
        ∂Y = add!!(∂Y, (Y * Y') * (ΔA' - (ΔA' * A) * Y))  # Y Y' ΔA' (I - A Y)
    end
    return Y, ∂Y
end

function rrule(
    ::typeof(pinv),
    x::AbstractVector{T},
    tol::Real = 0,
) where {T<:Union{Real,Complex}}
    y = pinv(x, tol)
    function pinv_pullback(Δy)
        ∂x = sum(abs2, parent(y)) .* vec(Δy') .- 2real(y * Δy') .* parent(y)
        return (NO_FIELDS, ∂x, Zero())
    end
    return y, pinv_pullback
end

function rrule(
    ::typeof(pinv),
    x::LinearAlgebra.AdjOrTransAbsVec{T},
    tol::Real = 0,
) where {T<:Union{Real,Complex}}
    y = pinv(x, tol)
    function pinv_pullback(Δy)
        ∂x′ = sum(abs2, y) .* Δy .- 2real(y' * Δy) .* y
        ∂x = x isa Transpose ? transpose(conj(∂x′)) : adjoint(∂x′)
        return (NO_FIELDS, ∂x, Zero())
    end
    return y, pinv_pullback
end

function rrule(::typeof(pinv), A::AbstractMatrix{T}; kwargs...) where {T}
    Y = pinv(A; kwargs...)
    function pinv_pullback(ΔY)
        m, n = size(A)
        # contract over the largest dimension
        if m ≤ n
            ∂A = (Y' * -ΔY) * Y'
            ∂A = add!!(∂A, (Y' * Y) * (ΔY' - (ΔY' * Y) * A)) # Y' Y ΔY' (I - Y A)
            ∂A = add!!(∂A, (I - A * Y) * (ΔY' * Y) * Y') # (I - A Y) ΔY' Y Y'
        elseif m > n
            ∂A = Y' * (-ΔY * Y')
            ∂A = add!!(∂A, Y' * (Y * ΔY') * (I - Y * A)) # Y' Y ΔY' (I - Y A)
            ∂A = add!!(∂A, (ΔY' - A * (Y * ΔY')) * (Y * Y')) # (I - A Y) ΔY' Y Y'
        end
        return (NO_FIELDS, ∂A)
    end
    return Y, pinv_pullback
end

#####
##### `norm`
#####

function frule((_, Δx, Δp), ::typeof(norm), x, p::Real)
    return if isempty(x)
        z = float(norm(zero(eltype(x))))
        z, zero(z * zero(Δp) + z * zero(eltype(Δx)))
    elseif p == 2
        _norm2_forward(x, Δx, Δp)
    elseif p == 1
        _norm1_forward(x, Δx, Δp)
    elseif p == Inf
        _normInf_forward(x, Δx, Δp; fnorm = LinearAlgebra.normInf)
    elseif p == 0
        z = typeof(float(norm(first(x))))(count(!iszero, x))
        z, zero(z * zero(Δp) + z * zero(first(Δx)))
    elseif p == -Inf
        _normInf_forward(x, Δx, Δp; fnorm = LinearAlgebra.normMinusInf)
    else
        _normp_forward(x, p, Δx, Δp)
    end
end
frule((Δself, Δx), ::typeof(norm), x) = frule((Δself, Δx, Zero()), norm, x, 2)
function frule((_, Δx), ::typeof(norm), x::Number, p::Real=2)
    y = norm(x, p)
    ∂y = if iszero(Δx) || iszero(p)
        zero(real(x)) * zero(real(Δx))
    else
        signx = x isa Real ? sign(x) : x * pinv(y)
        _realconjtimes(signx, Δx)
    end
    return y, ∂y
end
function frule(
    (Δself, Δx, Δp),
    ::typeof(norm),
    x::Union{LinearAlgebra.TransposeAbsVec,LinearAlgebra.AdjointAbsVec},
    p::Real,
)
    fdual = x isa Transpose ? transpose : adjoint
    return frule((Δself, vec(fdual(Δx)), Δp), norm, parent(x), p)
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
function rrule(
    ::typeof(norm),
    x::Union{StridedArray,LinearAlgebra.AbstractTriangular,Diagonal},
)
    y, inner_pullback = rrule(norm, x, 2)
    function norm_pullback(Δy)
        (∂self, ∂x) = inner_pullback(Δy)
        return (∂self, unthunk(∂x))
    end
    return y, norm_pullback
end
function rrule(::typeof(norm), x::Number, p::Real=2)
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

frule((_, Δx, Δp), ::typeof(LinearAlgebra.normp), x, p) = _normp_forward(x, p, Δx, Δp)

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

function _normp_forward(x, p, Δx, Δp = Zero())
    # TODO: accumulate `y` in parallel to `∂y`
    y = LinearAlgebra.normp(x, p)
    Δx isa AbstractZero && Δp isa AbstractZero && return (y, Zero())
    x_Δx = zip(x, Δx isa AbstractZero ? Iterators.repeated(Δx) : Δx)
    # non-differentiable wrt p at p ∈ {0, Inf}. use subgradient convention
    ∂logp = ifelse(iszero(p) || isinf(p), zero(Δp) / one(p), Δp / p)
    ((xi, Δxi), i) = iterate(x_Δx)::Tuple
    ∂y = zero(real(Δxi)) / one(y) + zero(y)
    y > 0 && isfinite(y) || return (y, zero(∂y))
    if !isa(∂logp, AbstractZero)
        ∂y -= y * log(y) * ∂logp
    end
    while true
        a = norm(xi)
        if !iszero(a)
            signxi = xi isa Real ? sign(xi) : xi / a
            ∂a = _realconjtimes(signxi, Δxi)
            n = (a / y)^(p - 1)
            ∂y += n * ∂a
            if !isa(∂logp, AbstractZero) && !iszero(a)
                ∂y += n * a * log(a) * ∂logp
            end
        end
        state = iterate(x_Δx, i)
        state === nothing && break
        ((xi, Δxi), i) = state
    end
    return y, ∂y
end

function _normp_back_x(x, p, y, Δy)
    Δu = real(Δy)
    ∂x = broadcast(x) do xi
        r = xi / y
        a = abs(r)
        ∂xi = r * a^(p - 2) * Δu
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

function frule((_, Δx), ::typeof(LinearAlgebra.normMinusInf), x)
    return _normInf_forward(x, Δx; fnorm = LinearAlgebra.normMinusInf)
end

function frule((_, Δx), ::typeof(LinearAlgebra.normInf), x)
    return _normInf_forward(x, Δx; fnorm = LinearAlgebra.normInf)
end

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

function _normInf_forward(x, Δx, Δp = Zero(); fnorm = LinearAlgebra.normInf)
    Δx isa AbstractZero && return (fnorm(x), Zero())
    x_Δx = zip(x, Δx)
    cmp = fnorm === LinearAlgebra.normInf ? (>) : (<)
    ((xi, Δxi), i) = iterate(x_Δx)::Tuple
    y = norm(xi)
    ∂y = _realconjtimes(sign(xi), Δxi)
    while true
        state = iterate(x_Δx, i)
        state === nothing && break
        ((xi, Δxi), i) = state
        a = norm(xi)
        # if multiple `xi`s have the exact same norm, then the corresponding `Δxi`s must
        # be identical if upstream rules behaved correctly, so any `Δxi` will do.
        (y, ∂y) = ifelse(
            isnan(y) | cmp(y, a),
            (y, ∂y),
            (a, _realconjtimes(sign(xi), Δxi)),
        )
    end
    return float(y), float(∂y)
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

frule((_, Δx), ::typeof(LinearAlgebra.norm1), x) = _norm1_forward(x, Δx)

function rrule(
    ::typeof(LinearAlgebra.norm1),
    x::Union{StridedArray,LinearAlgebra.AbstractTriangular,Diagonal},
)
    y = LinearAlgebra.norm1(x)
    norm1_pullback(Δy) = (NO_FIELDS, _norm1_back(x, y, Δy))
    norm1_pullback(::Zero) = (NO_FIELDS, Zero())
    return y, norm1_pullback
end

function _norm1_forward(x, Δx, Δp = Zero())
    Δx isa AbstractZero && return (LinearAlgebra.norm1(x), Zero())
    x_Δx = zip(x, Δx)
    ((xi, Δxi), i) = iterate(x_Δx)::Tuple
    a = float(norm(xi))
    T = typeof(a)
    y::promote_type(Float64, T) = a
    signxi = xi isa Real ? sign(xi) : xi / ifelse(iszero(a), one(a), a)
    ∂a = _realconjtimes(signxi, Δxi)
    T∂ = typeof(zero(∂a))
    ∂y::promote_type(Float64, T∂) = ∂a
    if !isa(Δp, AbstractZero) && !iszero(a)
        ∂y += a * log(a) * Δp
    end
    while true
        state = iterate(x_Δx, i)
        state === nothing && break
        ((xi, Δxi), i) = state
        a = norm(xi)
        y += a
        signxi = xi isa Real ? sign(xi) : xi / ifelse(iszero(a), one(a), a)
        ∂y += _realconjtimes(signxi, Δxi)
        if !isa(Δp, AbstractZero) && !iszero(a)
            ∂y += a * log(a) * Δp
        end
    end
    if !isa(Δp, AbstractZero) && !iszero(y) && isfinite(y)
        ∂y -= y * log(y) * Δp
    end
    return convert(T, y), convert(T∂, ∂y)
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
    ∂y = Δx isa AbstractZero ? Δx : real(dot(x, Δx)) * pinv(y)
    if !isa(Δp, AbstractZero) && !iszero(y)
        s = sum(x) do xi
            a = abs2(xi)
            return ifelse(iszero(a), zero(float(a)), a * log(a))
        end
        ∂y += (s / y / 2 - y * log(y)) * Δp / 2
    end
    return y, ∂y
end

_norm2_back(x, y, Δy) = x .* (real(Δy) * pinv(y))

#####
##### `normalize`/`normalize!`
#####

function frule((_, Δx, Δp), ::typeof(normalize!), x::AbstractVector, p::Real)
    (nrm, ∂nrm) = frule((Zero(), Δx, Δp), norm, x, p)
    _normalize_forward!(x, nrm, Δx, ∂nrm)
    return (x, Δx)
end
function frule((Δself, Δx), ::typeof(normalize!), x::AbstractVector)
    return frule((Δself, Δx, Zero()), normalize!, x, 2)
end

function frule((_, Δx, Δp), ::typeof(normalize), x::AbstractVector, p::Real)
    (nrm, ∂nrm) = frule((Zero(), Δx, Δp), norm, x, p)
    T = typeof(first(x) / nrm)
    y = copyto!(similar(x, T), x)
    ∂y = copyto!(similar(Δx, typeof(one(T) * ∂nrm + first(Δx))), Δx)
    _normalize_forward!(y, nrm, ∂y, ∂nrm)
    return y, ∂y
end
function frule((Δself, Δx), ::typeof(normalize), x::AbstractVector)
    return frule((Δself, Δx, Zero()), normalize, x, 2)
end

function rrule(::typeof(normalize), x::AbstractVector, p::Real)
    nrm, inner_pullback = rrule(norm, x, p)
    Ty = typeof(first(x) / nrm)
    y = copyto!(similar(x, Ty), x)
    LinearAlgebra.__normalize!(y, nrm)
    function normalize_pullback(Δy)
        invnrm = pinv(nrm)
        ∂nrm = -dot(y, Δy) * pinv(nrm)
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

function _normalize_forward!(x, nrm, Δx, Δnrm)
    LinearAlgebra.__normalize!(x, nrm)
    @inbounds for i in eachindex(x, Δx)
        Δx[i] -= x[i] * Δnrm
    end
    LinearAlgebra.__normalize!(Δx, nrm)
    return x, Δx
end
