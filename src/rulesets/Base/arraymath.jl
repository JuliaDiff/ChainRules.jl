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
