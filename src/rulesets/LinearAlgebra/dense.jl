using LinearAlgebra: AbstractTriangular

# Matrix wrapper types that we know are square and are thus potentially invertible. For
# these we can use simpler definitions for `/` and `\`.
const SquareMatrix{T} = Union{Diagonal{T},AbstractTriangular{T}}

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
##### `inv`
#####

function frule((_, Δx), ::typeof(inv), x::AbstractArray)
    Ω = inv(x)
    return Ω, -Ω * Δx * Ω
end

function rrule(::typeof(inv), x::AbstractArray)
    Ω = inv(x)
    function inv_pullback(ΔΩ)
        return NO_FIELDS, -Ω' * ΔΩ * Ω'
    end
    return Ω, inv_pullback
end

#####
##### `det`
#####

function frule((_, ẋ), ::typeof(det), x::Union{Number, AbstractMatrix})
    Ω = det(x)
    # TODO Performance optimization: probably there is an efficent
    # way to compute this trace without during the full compution within
    return Ω, Ω * tr(inv(x) * ẋ)
end

function rrule(::typeof(det), x::Union{Number, AbstractMatrix})
    Ω = det(x)
    function det_pullback(ΔΩ)
        return NO_FIELDS, Ω * ΔΩ * inv(x)'
    end
    return Ω, det_pullback
end

#####
##### `logdet`
#####

function frule((_, Δx), ::typeof(logdet), x::Union{Number, AbstractMatrix})
    Ω = logdet(x)
    return Ω, tr(inv(x) * Δx)
end

function rrule(::typeof(logdet), x::Union{Number, AbstractMatrix})
    Ω = logdet(x)
    function logdet_pullback(ΔΩ)
        return (NO_FIELDS, ΔΩ * inv(x)')
    end
    return Ω, logdet_pullback
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
        return (NO_FIELDS, @thunk Diagonal(fill(ΔΩ, size(x, 1))))
    end
    return tr(x), tr_pullback
end


#####
##### `*`
#####

function rrule(::typeof(*), A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})
    function times_pullback(Ȳ)
        return (NO_FIELDS, @thunk(Ȳ * B'), @thunk(A' * Ȳ))
    end
    return A * B, times_pullback
end

#####
##### `/`
#####

function rrule(::typeof(/), A::AbstractMatrix{<:Real}, B::T) where T<:SquareMatrix{<:Real}
    Y = A / B
    function slash_pullback(Ȳ)
        S = T.name.wrapper
        ∂A = @thunk Ȳ / B'
        ∂B = @thunk S(-Y' * (Ȳ / B'))
        return (NO_FIELDS, ∂A, ∂B)
    end
    return Y, slash_pullback
end

function rrule(::typeof(/), A::AbstractVecOrMat{<:Real}, B::AbstractVecOrMat{<:Real})
    Aᵀ, dA_pb = rrule(adjoint, A)
    Bᵀ, dB_pb = rrule(adjoint, B)
    Cᵀ, dS_pb = rrule(\, Bᵀ, Aᵀ)
    C, dC_pb = rrule(adjoint, Cᵀ)
    function slash_pullback(Ȳ)
        # Optimization note: dAᵀ, dBᵀ, dC are calculated no matter which partial you want
        _, dC = dC_pb(Ȳ)
        _, dBᵀ, dAᵀ = dS_pb(unthunk(dC))

        ∂A = last(dA_pb(unthunk(dAᵀ)))
        ∂B = last(dA_pb(unthunk(dBᵀ)))

        (NO_FIELDS, ∂A, ∂B)
    end
    return C, slash_pullback
end

#####
##### `\`
#####

function rrule(::typeof(\), A::T, B::AbstractVecOrMat{<:Real}) where T<:SquareMatrix{<:Real}
    Y = A \ B
    function backslash_pullback(Ȳ)
        S = T.name.wrapper
        ∂A = @thunk S(-(A' \ Ȳ) * Y')
        ∂B = @thunk A' \ Ȳ
        return NO_FIELDS, ∂A, ∂B
    end
    return Y, backslash_pullback
end

function rrule(::typeof(\), A::AbstractVecOrMat{<:Real}, B::AbstractVecOrMat{<:Real})
    Y = A \ B
    function backslash_pullback(Ȳ)
        ∂A = @thunk begin
            B̄ = A' \ Ȳ
            Ā = -B̄ * Y'
            _add!(Ā, (B - A * Y) * B̄' / A')
            _add!(Ā, A' \ Y * (Ȳ' - B̄'A))
            Ā
        end
        ∂B = @thunk A' \ Ȳ
        return NO_FIELDS, ∂A, ∂B
    end
    return Y, backslash_pullback

end

#####
##### `norm`
#####

function rrule(::typeof(norm), A::AbstractArray{<:Real}, p::Real=2)
    y = norm(A, p)
    function norm_pullback(ȳ)
        u = y^(1-p)
        ∂A = @thunk ȳ .* u .* abs.(A).^p ./ A
        ∂p = @thunk ȳ * (u * sum(a->abs(a)^p * log(abs(a)), A) - y * log(y)) / p
        (NO_FIELDS, ∂A, ∂p)
    end
    return y, norm_pullback
end

function rrule(::typeof(norm), x::Real, p::Real=2)
    function norm_pullback(ȳ)
        ∂x = @thunk ȳ * sign(x)
        ∂p = @thunk zero(x)  # TODO: should this be Zero()?
        (NO_FIELDS, ∂x, ∂p)
    end
    return norm(x, p), norm_pullback
end

#####
##### `normp`
#####

function frule((_, Δx, Δp), ::typeof(LinearAlgebra.normp), x, p)
    # TODO: accumulate `y` in parallel to `∂y`
    y = LinearAlgebra.normp(x, p)
    x_Δx = zip(x, Δx isa AbstractZero ? Iterators.repeated(Δx) : Δx)
    # non-differentiable wrt p at p ∈ {0, Inf}. use subgradient convention
    ∂logp = ifelse(iszero(p) || isinf(p), zero(Δp) / one(p), Δp / p)
    ((xi, Δxi), i) = iterate(x_Δx)::Tuple
    ∂y = zero(real(Δxi)) / one(y) - (∂logp isa AbstractZero ? zero(y) : y * log(y) * ∂logp)
    iszero(y) || isinf(y) && return (y, zero(∂y))
    while true
        a = norm(xi)
        if !iszero(a)
            signxi = xi isa Real ? sign(xi) : xi / a
            ∂a = _realconjtimes(signxi, Δxi)
            ∂y += (a / y)^(p - 1) * (∂logp isa AbstractZero ? ∂a : ∂a + a * log(a) * ∂logp)
        end
        state = iterate(x_Δx, i)
        state === nothing && break
        ((xi, Δxi), i) = state
    end
    return y, ∂y
end

function rrule(::typeof(LinearAlgebra.normp), x, p)
    y = LinearAlgebra.normp(x, p)
    normp_pullback(Δy) = (NO_FIELDS, _normp_back(x, p, y, Δy)...)
    return y, normp_pullback
end

@inline function _normp_back(x, p, y, Δy)
    Δy = real(Δy)
    ∂x = @thunk broadcast(x) do xi
        a = abs(xi)
        signxi = xi isa Real ? sign(xi) : xi / a
        ∂xi = signxi * (a / y)^(p - 1) * Δy
        return ifelse(isfinite(∂xi), ∂xi, zero(∂xi))
    end
    ∂p = Thunk() do
        s = sum(x) do xi
            a = norm(xi)
            return (a / y)^p * log(ifelse(iszero(a), one(a), a))
        end
        p̄ = y * (s - log(y)) / p
        return ifelse(isfinite(p̄), p̄, zero(p̄))
    end
    return ∂x, ∂p
end

#####
##### `normMinusInf`/`normInf`
#####

function frule(
    (_, Δx),
    fnorm::Union{typeof(LinearAlgebra.normMinusInf),typeof(LinearAlgebra.normInf)},
    x,
)
    Δx isa AbstractZero && return (fnorm(x), Zero())
    x_Δx = zip(x, Δx)
    fcmp = fnorm === LinearAlgebra.normMinusInf ? (<) : (>)
    ((xi, Δxi), i) = iterate(x_Δx)::Tuple
    y = norm(xi)
    ∂y = _realconjtimes(sign(xi), Δxi)
    while true
        state = iterate(x_Δx, i)
        state === nothing && break
        ((xi, Δxi), i) = state
        a = norm(xi)
        (y, ∂y) = ifelse(
            isnan(y) | fcmp(y, a),
            (y, ∂y),
            (a, _realconjtimes(sign(xi), Δxi)),
        )
    end

    return float(y), float(∂y)
end

function rrule(::typeof(LinearAlgebra.normMinusInf), x)
    y = LinearAlgebra.normMinusInf(x)
    normMinusInf_pullback(Δy) = (NO_FIELDS, _normInf_back(x, y, Δy))
    return y, normMinusInf_pullback
end

function rrule(::typeof(LinearAlgebra.normInf), x)
    y = LinearAlgebra.normInf(x)
    normInf_pullback(Δy) = (NO_FIELDS, _normInf_back(x, y, Δy))
    return y, normInf_pullback
end

@inline function _normInf_back(x, y, Δy)
    # if multiple `xi`s have the exact same norm, then they must have been identically
    # produced, e.g. with `fill`. So we set only one to be non-zero.
    # we choose last index to match the `frule`.
    yind = findlast(xi -> norm(xi) == y, x)
    yind === nothing && throw(ArgumentError("y is not the correct norm of x"))
    Δu = real(Δy)
    ∂x = broadcast(enumerate(x)) do (i, xi)
        i == yind ? sign(xi) * Δu : zero(float(xi)) * zero(Δu)
    end
    return ∂x
end

#####
##### `norm1`
#####

function frule((_, Δx), ::typeof(LinearAlgebra.norm1), x)
    Δx isa AbstractZero && return (LinearAlgebra.norm1(x), Zero())
    x_Δx = zip(x, Δx)
    ((xi, Δxi), i) = iterate(x_Δx)::Tuple
    a = float(norm(xi))
    T = typeof(a)
    y::promote_type(Float64, T) = a
    ∂a = _realconjtimes(xi isa Real ? sign(xi) : xi / a, Δxi)
    T∂ = typeof(zero(∂a))
    ∂y::promote_type(Float64, T∂) = ∂a
    while true
        state = iterate(x_Δx, i)
        state === nothing && break
        ((xi, Δxi), i) = state
        a = norm(xi)
        y += a
        ∂y += _realconjtimes(xi isa Real ? sign(xi) : xi / a, Δxi)
    end
    return convert(T, y), convert(T∂, ∂y)
end

function rrule(::typeof(LinearAlgebra.norm1), x)
    y = LinearAlgebra.norm1(x)
    norm1_pullback(Δy) = (NO_FIELDS, _norm1_back(x, y, Δy))
    return y, norm1_pullback
end

@inline _norm1_back(x, y, Δy) = sign.(x) .* real(Δy)

#####
##### `norm2`
#####

function frule((_, Δx), ::typeof(LinearAlgebra.norm2), x)
    y = LinearAlgebra.norm2(x)
    # since dot product is efficient for pushforward, we don't accumulate in parallel
    n = ifelse(iszero(y), zero(y), y)
    ∂y = Δx isa AbstractZero ? Zero() : real(dot(x, Δx)) / n
    return y, ∂y
end

function rrule(::typeof(LinearAlgebra.norm2), x)
    y = LinearAlgebra.norm2(x)
    norm2_pullback(Δy) = (NO_FIELDS, _norm2_back(x, y, Δy))
    return y, norm2_pullback
end

@inline function _norm2_back(x, y, Δy)
    n = ifelse(iszero(y), zero(y), y)
    return _realconjtimes.(x, real(Δy)) ./ n
end
