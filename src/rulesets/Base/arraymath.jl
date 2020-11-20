######
###### `inv`
######

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
##### `*`
#####

function rrule(
    ::typeof(*),
    A::AbstractVecOrMat{<:CommutativeMulNumber},
    B::AbstractVecOrMat{<:CommutativeMulNumber},
)
    function times_pullback(Ȳ)
        return (
            NO_FIELDS,
            InplaceableThunk(
                @thunk(Ȳ * B'),
                X̄ -> mul!(X̄, Ȳ, B', true, true)
            ),
            InplaceableThunk(
                @thunk(A' * Ȳ),
                X̄ -> mul!(X̄, A', Ȳ, true, true)
            )
        )
    end
    return A * B, times_pullback
end

function rrule(
    ::typeof(*),
    A::AbstractVector{<:CommutativeMulNumber},
    B::AbstractMatrix{<:CommutativeMulNumber},
)
    function times_pullback(Ȳ)
        @assert size(B, 1) === 1   # otherwise primal would have failed.
        return (
            NO_FIELDS,
            InplaceableThunk(
                @thunk(Ȳ * vec(B')),
                X̄ -> mul!(X̄, Ȳ, vec(B'), true, true)
            ),
            InplaceableThunk(
                @thunk(A' * Ȳ),
                X̄ -> mul!(X̄, A', Ȳ, true, true)
            )
        )
    end
    return A * B, times_pullback
end

function rrule(
   ::typeof(*), A::CommutativeMulNumber, B::AbstractArray{<:CommutativeMulNumber}
)
    function times_pullback(Ȳ)
        return (
            NO_FIELDS,
            @thunk(dot(Ȳ, B)'),
            InplaceableThunk(
                @thunk(A' * Ȳ),
                X̄ -> mul!(X̄, conj(A), Ȳ, true, true)
            )
        )
    end
    return A * B, times_pullback
end

function rrule(
    ::typeof(*), B::AbstractArray{<:CommutativeMulNumber}, A::CommutativeMulNumber
)
    function times_pullback(Ȳ)
        return (
            NO_FIELDS,
            InplaceableThunk(
                @thunk(A' * Ȳ),
                X̄ -> mul!(X̄, conj(A), Ȳ, true, true)
            ),
            @thunk(dot(Ȳ, B)'),
        )
    end
    return A * B, times_pullback
end


#####
##### `muladd`
#####

function rrule(
        ::typeof(muladd),
        A::AbstractMatrix{<:CommutativeMulNumber},
        B::AbstractVecOrMat{<:CommutativeMulNumber},
        z::Union{CommutativeMulNumber, AbstractVecOrMat{<:CommutativeMulNumber}},
    )
    # The useful case, mul! fused with +
    function muladd_pullback_1(Ȳ)
        matmul = (
            InplaceableThunk(
                @thunk(Ȳ * B'),
                dA -> mul!(dA, Ȳ, B', true, true)
            ),
            InplaceableThunk(
                @thunk(A' * Ȳ),
                dB -> mul!(dB, A', Ȳ, true, true)
            )
        )
        addon = if z isa Bool
            Zero()
        elseif z isa Number
            @thunk(sum(Ȳ))
        else
            T = eltype(Ȳ)
            InplaceableThunk(
                @thunk(sum!(similar(z, T), Ȳ)),
                dz -> sum!(dz, Ȳ; init=false)
            )
        end
        (NO_FIELDS, matmul..., addon)
    end
    return muladd(A, B, z), muladd_pullback_1
end

function rrule(
        ::typeof(muladd),
        ut::LinearAlgebra.AdjOrTransAbsVec{<:CommutativeMulNumber},
        v::AbstractVector{<:CommutativeMulNumber},
        z::CommutativeMulNumber,
    )
    # This case is dot(u,v)+z, but would also match signature above.
    muladd_pullback_2(dy) = (NO_FIELDS, @thunk(v' .* dy), @thunk(ut' .* dy), dy)
    return muladd(ut, v, z), muladd_pullback_2
end

function rrule(
        ::typeof(muladd),
        u::AbstractVector{<:CommutativeMulNumber},
        vt::LinearAlgebra.AdjOrTransAbsVec{<:CommutativeMulNumber},
        z::Union{CommutativeMulNumber, AbstractVecOrMat{<:CommutativeMulNumber}},
    )
    # Outer product, just broadcasting
    function muladd_pullback_3(Ȳ)
        proj = (
            @thunk(vec(sum(Ȳ .* conj.(vt), dims=2))),
            @thunk(vec(sum(u .* conj.(Ȳ), dims=1))'),
        )
        addon = if z isa Bool
            Zero()
        elseif z isa Number
            @thunk(sum(Ȳ))
        else
            T = eltype(Ȳ)
            InplaceableThunk(
                @thunk(sum!(similar(z, T), Ȳ)),
                dz -> sum!(dz, Ȳ; init=false)
            )
        end
        (NO_FIELDS, proj..., addon)
    end
    return muladd(u, vt, z), muladd_pullback_3
end


#####
##### `/`
#####

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

function rrule(::typeof(\), A::AbstractVecOrMat{<:Real}, B::AbstractVecOrMat{<:Real})
    Y = A \ B
    function backslash_pullback(Ȳ)
        ∂A = @thunk begin
            B̄ = A' \ Ȳ
            Ā = -B̄ * Y'
            Ā = add!!(Ā, (B - A * Y) * B̄' / A')
            Ā = add!!(Ā, A' \ Y * (Ȳ' - B̄'A))
            Ā
        end
        ∂B = @thunk A' \ Ȳ
        return NO_FIELDS, ∂A, ∂B
    end
    return Y, backslash_pullback

end

#####
##### `\`, `/` matrix-scalar_rule
#####

function rrule(::typeof(/), A::AbstractArray{<:Real}, b::Real)
    Y = A/b
    function slash_pullback(Ȳ)
        return (NO_FIELDS, @thunk(Ȳ/b), @thunk(-dot(Ȳ, Y)/b))
    end
    return Y, slash_pullback
end

function rrule(::typeof(\), b::Real, A::AbstractArray{<:Real})
    Y = b\A
    function backslash_pullback(Ȳ)
        return (NO_FIELDS, @thunk(-dot(Ȳ, Y)/b), @thunk(Ȳ/b))
    end
    return Y, backslash_pullback
end

#####
##### Negation (Unary -)
#####

function rrule(::typeof(-), x::AbstractArray)
    function negation_pullback(ȳ)
        return NO_FIELDS, InplaceableThunk(@thunk(-ȳ), ā -> ā .-= ȳ)
    end
    return -x, negation_pullback
end
