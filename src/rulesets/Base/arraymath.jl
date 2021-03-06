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
        return NoTangent(), -Ω' * ΔΩ * Ω'
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
    function times_pullback(ȳ)
        Ȳ = unthunk(ȳ)
        return (
            NoTangent(),
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
    function times_pullback(ȳ)
        Ȳ = unthunk(ȳ)
        @assert size(B, 1) === 1   # otherwise primal would have failed.
        return (
            NoTangent(),
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
    function times_pullback(ȳ)
        Ȳ = unthunk(ȳ)
        return (
            NoTangent(),
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
    function times_pullback(ȳ)
        Ȳ = unthunk(ȳ)
        return (
            NoTangent(),
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
    function muladd_pullback_1(ȳ)
        Ȳ = unthunk(ȳ)
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
            NoTangent()
        elseif z isa Number
            @thunk(sum(Ȳ))
        else
            InplaceableThunk(
                @thunk(sum!(similar(z, eltype(Ȳ)), Ȳ)),
                dz -> sum!(dz, Ȳ; init=false)
            )
        end
        (NoTangent(), matmul..., addon)
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
    function muladd_pullback_2(ȳ)
        dy = unthunk(ȳ)
        ut_thunk = InplaceableThunk(
            @thunk(v' .* dy),
            dut -> dut .+= v' .* dy
        )
        v_thunk = InplaceableThunk(
            @thunk(ut' .* dy),
            dv -> dv .+= ut' .* dy
        )
        (NoTangent(), ut_thunk, v_thunk, z isa Bool ? NoTangent() : dy)
    end
    return muladd(ut, v, z), muladd_pullback_2
end

function rrule(
        ::typeof(muladd),
        u::AbstractVector{<:CommutativeMulNumber},
        vt::LinearAlgebra.AdjOrTransAbsVec{<:CommutativeMulNumber},
        z::Union{CommutativeMulNumber, AbstractVecOrMat{<:CommutativeMulNumber}},
    )
    # Outer product, just broadcasting
    function muladd_pullback_3(ȳ)
        Ȳ = unthunk(ȳ)
        proj = (
            @thunk(vec(sum(Ȳ .* conj.(vt), dims=2))),
            @thunk(vec(sum(u .* conj.(Ȳ), dims=1))'),
        )
        addon = if z isa Bool
            NoTangent()
        elseif z isa Number
            @thunk(sum(Ȳ))
        else
            InplaceableThunk(
                @thunk(sum!(similar(z, eltype(Ȳ)), Ȳ)),
                dz -> sum!(dz, Ȳ; init=false)
            )
        end
        (NoTangent(), proj..., addon)
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

        (NoTangent(), ∂A, ∂B)
    end
    return C, slash_pullback
end

#####
##### `\`
#####

function rrule(::typeof(\), A::AbstractVecOrMat{<:Real}, B::AbstractVecOrMat{<:Real})
    Y = A \ B
    function backslash_pullback(ȳ)
        Ȳ = unthunk(ȳ)
        ∂A = @thunk begin
            B̄ = A' \ Ȳ
            Ā = -B̄ * Y'
            Ā = add!!(Ā, (B - A * Y) * B̄' / A')
            Ā = add!!(Ā, A' \ Y * (Ȳ' - B̄'A))
            Ā
        end
        ∂B = @thunk A' \ Ȳ
        return NoTangent(), ∂A, ∂B
    end
    return Y, backslash_pullback

end

#####
##### `\`, `/` matrix-scalar_rule
#####

function rrule(::typeof(/), A::AbstractArray{<:CommutativeMulNumber}, b::CommutativeMulNumber)
    Y = A/b
    function slash_pullback_scalar(ȳ)
        Ȳ = unthunk(ȳ)
        Athunk = InplaceableThunk(
            @thunk(Ȳ / conj(b)),
            dA -> dA .+= Ȳ ./ conj(b),
        )
        bthunk = @thunk(-dot(A,Ȳ) / conj(b^2))
        return (NoTangent(), Athunk, bthunk)
    end
    return Y, slash_pullback_scalar
end

function rrule(::typeof(\), b::CommutativeMulNumber, A::AbstractArray{<:CommutativeMulNumber})
    Y, back = rrule(/, A, b)
    function backslash_pullback(dY)  # just reverses the arguments!
        d0, dA, db = back(dY)
        return (d0, db, dA)
    end
    return Y, backslash_pullback
end

#####
##### Negation (Unary -)
#####

function rrule(::typeof(-), x::AbstractArray)
    function negation_pullback(ȳ)
        return NoTangent(), InplaceableThunk(@thunk(-ȳ), ā -> ā .-= ȳ)
    end
    return -x, negation_pullback
end


#####
##### Addition (Multiarg `+`)
#####

function rrule(::typeof(+), arrs::AbstractArray...)
    y = +(arrs...)
    arr_axs = map(axes, arrs)
    function add_pullback(dy)
        return (NoTangent(), map(ax -> reshape(dy, ax), arr_axs)...)
    end
    return y, add_pullback
end
