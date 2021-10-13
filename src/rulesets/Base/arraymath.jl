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
        return NoTangent(), Ω' * -ΔΩ * Ω'
    end
    return Ω, inv_pullback
end

#####
##### `*`
#####

function rrule(::typeof(*), A::AbstractVecOrMat{<:Number}, B::AbstractVecOrMat{<:Number})
    project_A = ProjectTo(A)
    project_B = ProjectTo(B)
    function times_pullback(ȳ)
        Ȳ = unthunk(ȳ)
        dA = @thunk(project_A(Ȳ * B'))
        dB = @thunk(project_B(A' * Ȳ))
        return NoTangent(), dA, dB
    end
    return A * B, times_pullback
end

# Optimized case for StridedMatrixes
# no need to project as already dense, and we are allowed to use InplaceableThunk because
# we know the destination will also be dense. TODO workout how to apply this generally:
# https://github.com/JuliaDiff/ChainRulesCore.jl/issues/411
function rrule(
    ::typeof(*),
    A::StridedMatrix{<:CommutativeMulNumber},
    B::StridedVecOrMat{<:CommutativeMulNumber},
)
    function times_pullback(ȳ)
        Ȳ = unthunk(ȳ)
        dA = InplaceableThunk(
            X̄ -> mul!(X̄, Ȳ, B', true, true),
            @thunk(Ȳ * B'),
        )
        dB = InplaceableThunk(
            X̄ -> mul!(X̄, A', Ȳ, true, true),
            @thunk(A' * Ȳ),
        )
        return NoTangent(), dA, dB
    end
    return A * B, times_pullback
end

function rrule(::typeof(*), A::AbstractVector{<:Number}, B::AbstractMatrix{<:Number})
    project_A = ProjectTo(A)
    project_B = ProjectTo(B)
    function times_pullback(ȳ)
        Ȳ = unthunk(ȳ)
        @assert size(B, 1) === 1   # otherwise primal would have failed.
        return (
            NoTangent(),
            InplaceableThunk(
                X̄ -> mul!(X̄, Ȳ, vec(B'), true, true),
                @thunk(project_A(Ȳ * vec(B'))),
            ),
            InplaceableThunk(
                X̄ -> mul!(X̄, A', Ȳ, true, true),
                @thunk(project_B(A' * Ȳ)),
            )
        )
    end
    return A * B, times_pullback
end

function rrule(::typeof(*), A::Number, B::AbstractArray{<:Number})
    project_A = ProjectTo(A)
    project_B = ProjectTo(B)
    function times_pullback(ȳ)
        Ȳ = unthunk(ȳ)
        return (
            NoTangent(),
            @thunk(project_A(eltype(B) isa CommutativeMulNumber ? dot(Ȳ, B)' : dot(Ȳ', B'))),
            InplaceableThunk(
                X̄ -> mul!(X̄, conj(A), Ȳ, true, true),
                @thunk(project_B(A' * Ȳ)),
            )
        )
    end
    return A * B, times_pullback
end

function rrule(
    ::typeof(*), B::AbstractArray{<:Number}, A::Number
)
    project_A = ProjectTo(A)
    project_B = ProjectTo(B)
    function times_pullback(ȳ)
        Ȳ = unthunk(ȳ)
        return (
            NoTangent(),
            InplaceableThunk(
                X̄ -> mul!(X̄, conj(A), Ȳ, true, true),
                @thunk(project_B(A' * Ȳ)),
            ),
            @thunk(project_A(eltype(A) isa CommutativeMulNumber ? dot(Ȳ, B)' : dot(Ȳ', B'))),
        )
    end
    return A * B, times_pullback
end


#####
##### `muladd`
#####

function rrule(
        ::typeof(muladd),
        A::AbstractMatrix{<:Number},
        B::AbstractVecOrMat{<:Number},
        z::Union{Number, AbstractVecOrMat{<:Number}},
    )
    project_A = ProjectTo(A)
    project_B = ProjectTo(B)
    project_z = ProjectTo(z)

    # The useful case, mul! fused with +
    function muladd_pullback_1(ȳ)
        Ȳ = unthunk(ȳ)
        matmul = (
            InplaceableThunk(
                dA -> mul!(dA, Ȳ, B', true, true),
                @thunk(project_A(Ȳ * B')),
            ),
            InplaceableThunk(
                dB -> mul!(dB, A', Ȳ, true, true),
                @thunk(project_B(A' * Ȳ)),
            )
        )
        addon = if z isa Bool
            NoTangent()
        elseif z isa Number
            @thunk(project_z(sum(Ȳ)))
        else
            InplaceableThunk(
                dz -> sum!(dz, Ȳ; init=false),
                @thunk(project_z(sum!(similar(z, eltype(Ȳ)), Ȳ))),
            )
        end
        (NoTangent(), matmul..., addon)
    end
    return muladd(A, B, z), muladd_pullback_1
end

function rrule(
        ::typeof(muladd),
        ut::LinearAlgebra.AdjOrTransAbsVec{<:Number},
        v::AbstractVector{<:Number},
        z::Number,
    )
    project_ut = ProjectTo(ut)
    project_v = ProjectTo(v)
    project_z = ProjectTo(z)

    # This case is dot(u,v)+z, but would also match signature above.
    function muladd_pullback_2(ȳ)
        dy = unthunk(ȳ)
        ut_thunk = InplaceableThunk(
            dut -> dut .+= v' .* dy,
            @thunk(project_ut((v * dy')')),
        )
        v_thunk = InplaceableThunk(
            dv -> dv .+= ut' .* dy,
            @thunk(project_v(ut' * dy)),
        )
        (NoTangent(), ut_thunk, v_thunk, z isa Bool ? NoTangent() : project_z(dy))
    end
    return muladd(ut, v, z), muladd_pullback_2
end

function rrule(
        ::typeof(muladd),
        u::AbstractVector{<:Number},
        vt::LinearAlgebra.AdjOrTransAbsVec{<:Number},
        z::Union{Number, AbstractVecOrMat{<:Number}},
    )
    project_u = ProjectTo(u)
    project_vt = ProjectTo(vt)
    project_z = ProjectTo(z)

    # Outer product, just broadcasting
    function muladd_pullback_3(ȳ)
        Ȳ = unthunk(ȳ)
        proj = (
            @thunk(project_u(Ȳ * vec(vt'))),
            @thunk(project_vt((Ȳ' * u)')),
        )
        addon = if z isa Bool
            NoTangent()
        elseif z isa Number
            @thunk(project_z(sum(Ȳ)))
        else
            InplaceableThunk(
                dz -> sum!(dz, Ȳ; init=false),
                @thunk(project_z(sum!(similar(z, eltype(Ȳ)), Ȳ))),
            )
        end
        (NoTangent(), proj..., addon)
    end
    return muladd(u, vt, z), muladd_pullback_3
end

#####
##### `/`
#####

function rrule(::typeof(/), A::AbstractVecOrMat{<:Number}, B::AbstractVecOrMat{<:Number})
    C = A / B
    project_A = ProjectTo(A)
    project_B = ProjectTo(B)
    function slash_pullback(ΔC)
        ∂A = unthunk(ΔC) / B'
        ∂B = InplaceableThunk(
            @thunk(C' * -∂A),
            B̄ -> mul!(B̄, C', ∂A, -1, true)
        )
        return NoTangent(), project_A(∂A), project_B(∂B)
    end
    return C, slash_pullback
end

#####
##### `\`
#####

function rrule(::typeof(\), A::AbstractVecOrMat{<:Real}, B::AbstractVecOrMat{<:Real})
    project_A = ProjectTo(A)
    project_B = ProjectTo(B)

    Y = A \ B
    function backslash_pullback(ȳ)
        Ȳ = unthunk(ȳ)
        ∂A = @thunk begin
            B̄ = A' \ Ȳ
            Ā = -B̄ * Y'
            Ā = add!!(Ā, (B - A * Y) * B̄' / A')
            Ā = add!!(Ā, A' \ Y * (Ȳ' - B̄'A))
            project_A(Ā)
        end
        ∂B = @thunk project_B(A' \ Ȳ)
        return NoTangent(), ∂A, ∂B
    end
    return Y, backslash_pullback

end

#####
##### `\`, `/` matrix-scalar_rule
#####

function rrule(::typeof(/), A::AbstractArray{<:Number}, b::Number)
    Y = A/b
    function slash_pullback_scalar(ȳ)
        Ȳ = unthunk(ȳ)
        Athunk = InplaceableThunk(
            dA -> dA .+= Ȳ ./ conj(b),
            @thunk(Ȳ / conj(b)),
        )
        bthunk = @thunk(-dot(Y,Ȳ) / conj(b))
        return (NoTangent(), Athunk, bthunk)
    end
    return Y, slash_pullback_scalar
end

function rrule(::typeof(\), b::Number, A::AbstractArray{<:Number})
    Y = b \ A
    function backslash_pullback(ȳ)
        Ȳ = unthunk(ȳ)
        Athunk = InplaceableThunk(
            @thunk(conj(b) \ Ȳ),
            dA -> dA .+= conj(b) .\ Ȳ,
        )
        bthunk = if eltype(Y) isa CommutativeMulNumber
            @thunk(-conj(b) \ dot(Y, Ȳ))
        else
            @thunk(-conj(b) \ dot(Ȳ',Y'))
        end
        return (NoTangent(), Athunk, bthunk)
    end
    return Y, backslash_pullback
end

#####
##### Negation (Unary -)
#####

function rrule(::typeof(-), x::AbstractArray)
    function negation_pullback(ȳ)
        return NoTangent(), InplaceableThunk(ā -> ā .-= ȳ, @thunk(-ȳ))
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
