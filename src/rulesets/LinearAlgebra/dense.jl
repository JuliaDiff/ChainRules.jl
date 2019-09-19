using LinearAlgebra: AbstractTriangular

# Matrix wrapper types that we know are square and are thus potentially invertible. For
# these we can use simpler definitions for `/` and `\`.
const SquareMatrix{T} = Union{Diagonal{T},AbstractTriangular{T}}

#####
##### `dot`
#####

function frule(::typeof(dot), x, y)
    function dot_pushforward(Δself, Δx, Δy)
        return sum(Δx * cast(y)) + sum(cast(x) * Δy)
    end
    return dot(x, y), dot_pushforward
end

function rrule(::typeof(dot), x, y)
    function dot_pullback(ΔΩ)
        return (NO_FIELDS, ΔΩ * cast(y), cast(x) * ΔΩ,)
    end
    return dot(x, y), dot_pullback
end

#####
##### `inv`
#####

function frule(::typeof(inv), x::AbstractArray)
    Ω = inv(x)
    m = @thunk(-Ω)
    function inv_pushforward(_, Δx)
        return m * Δx * Ω
    end
    return Ω, inv_pushforward
end

function rrule(::typeof(inv), x::AbstractArray)
    Ω = inv(x)
    m = @thunk(-Ω')
    function inv_pullback(ΔΩ)
        return NO_FIELDS, m * ΔΩ * Ω'
    end
    return Ω, inv_pullback
end

#####
##### `det`
#####

function frule(::typeof(det), x)
    Ω = det(x)
    function det_pushforward(_, ẋ)
        # TODO Performance optimization: probably there is an efficent
        # way to compute this trace without during the full compution within
        return Ω * tr(inv(x) * ẋ)
    end
    return Ω, det_pushforward
end

function rrule(::typeof(det), x)
    Ω = det(x)
    function det_pullback(ΔΩ)
        return NO_FIELDS, @thunk(Ω * ΔΩ * inv(x)')
    end
    return Ω, det_pullback
end

#####
##### `logdet`
#####

function frule(::typeof(logdet), x)
    Ω = logdet(x)
    function logdet_pushforward(_, Δx)
        return tr(inv(x) * Δx)
    end
    return Ω, logdet_pushforward
end

function rrule(::typeof(logdet), x)
    Ω = logdet(x)
    function logdet_pullback(ΔΩ)
        return (NO_FIELDS, @thunk(ΔΩ * inv(x)'))
    end
    return Ω, logdet_pullback
end

#####
##### `trace`
#####

function frule(::typeof(tr), x)
    function tr_pushforward(_, Δx)
        return tr(Δx)
    end
    return tr(x), tr_pushforward
end

function rrule(::typeof(tr), x)
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
        # this is not a problem if you want the 2nd or 3rd, but if you want the first, it
        # is fairly wasteful
        _, dC = dC_pb(Ȳ)
        _, dBᵀ, dAᵀ = dS_pb(extern(dC))

        # need to extern as  dAᵀ, dBᵀ  are generally `Thunk`s, which don't support adjoint
        ∂A = @thunk last(dA_pb(extern(dAᵀ)))
        ∂B = @thunk last(dA_pb(extern(dBᵀ)))

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
