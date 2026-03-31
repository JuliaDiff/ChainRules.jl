#####
##### Hermitian/Symmetric sparse matrices
#####

const HermSparse{T, I} = Hermitian{T, SparseMatrixCSC{T, I}}
const SymSparse{T, I} = Symmetric{T, SparseMatrixCSC{T, I}}
const HermOrSymSparse{T, I} = Union{HermSparse{T, I}, SymSparse{T, I}}

const DenseMat{T} = Union{StridedMatrix{T}, AdjOrTrans{T, <:StridedVecOrMat{T}}}
const DenseVecOrMat{T} = Union{DenseMat{T}, StridedVector{T}}

function unwrap(A)
    if A isa Adjoint
        B = parent(A)

        if B isa Transpose
            return (parent(B), Val(:N), Val(:C))
        else
            return (B,         Val(:T), Val(:C))
        end
    elseif A isa Transpose
        B = parent(A)

        if B isa Adjoint
            return (parent(B), Val(:N), Val(:C))
        else
            return (B,         Val(:T), Val(:N))
        end
    else
        return (A, Val(:N), Val(:N))
    end
end

#####
##### selupd!
#####

# SELected UPDate: compute the selected low-rank update
#
#   C ← α A Bᴴ + conj(α) B Aᴴ + β C
#
# The update is only applied to the structural nonzeros of C.
function selupd!(C::HermSparse, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
    selupd!(parent(C), C.uplo, A, adjoint(B),      α,  β)
    selupd!(parent(C), C.uplo, B, adjoint(A), conj(α), 1)
    return C
end

# SELected UPDate: compute the selected low-rank update
#
#   C ← α A Bᴴ + α conj(B) Aᵀ + β C
#
# The update is only applied to the structural nonzeros of C.
function selupd!(C::SymSparse, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
    selupd!(parent(C), C.uplo, A,                     adjoint(B),   α, β)
    selupd!(parent(C), C.uplo, adjoint(transpose(B)), transpose(A), α, 1)
    return C
end

# SELected UPDate: compute the selected low-rank update
#
#   C ← α A B + β C
#
# The update is only applied to the structural nonzeros of C.
function selupd!(C::SparseMatrixCSC, uplo::Char, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
    AP, tA, cA = unwrap(A)
    BP, tB, cB = unwrap(B)
    return selupd_impl!(C, uplo, AP, BP, α, β, tA, cA, tB, cB)
end

function selupd_impl!(C::SparseMatrixCSC, uplo::Char, A::AbstractVector, B::AbstractVector, α, β, ::Val{tA}, ::Val{cA}, ::Val{tB}, ::Val{cB}) where {tA, cA, tB, cB}
    @assert size(C, 1) == size(C, 2) == length(A) == length(B)

    @inbounds for j in axes(C, 2)
        Bj = cB === :C ? conj(B[j]) : B[j]

        for p in nzrange(C, j)
            i = rowvals(C)[p]

            if (uplo == 'L' && i >= j) || (uplo == 'U' && i <= j)
                Ai = cA === :C ? conj(A[i]) : A[i]

                if iszero(β)
                    nonzeros(C)[p] = α * Ai * Bj
                else
                    nonzeros(C)[p] = β * nonzeros(C)[p] + α * Ai * Bj
                end
            end
        end
    end

    return C
end

function selupd_impl!(C::SparseMatrixCSC, uplo::Char, A::AbstractMatrix, B::AbstractMatrix, α, β, tA::Val{TA}, cA::Val{CA}, tB::Val{TB}, cB::Val{CB}) where {TA, CA, TB, CB}
    @assert size(C, 1) == size(C, 2)

    if TA === :N && TB === :N
        @assert size(A, 1) == size(C, 1)
        @assert size(B, 2) == size(C, 1)
        @assert size(A, 2) == size(B, 1)
    elseif TA === :N && TB !== :N
        @assert size(A, 1) == size(C, 1)
        @assert size(B, 1) == size(C, 1)
        @assert size(A, 2) == size(B, 2)
    elseif TA !== :N && TB === :N
        @assert size(A, 2) == size(C, 1)
        @assert size(B, 2) == size(C, 1)
        @assert size(A, 1) == size(B, 1)
    else
        @assert size(A, 2) == size(C, 1)
        @assert size(B, 1) == size(C, 1)
        @assert size(A, 1) == size(B, 2)
    end

    if TA === :N
        rng = axes(A, 2)
    else
        rng = axes(A, 1)
    end

    if iszero(β)
        fill!(nonzeros(C), β)
    else
        rmul!(nonzeros(C), β)
    end

    for k in rng
        if TA === :N
            Ak = view(A, :, k)
        else
            Ak = view(A, k, :)
        end

        if TB === :N
            Bk = view(B, k, :)
        else
            Bk = view(B, :, k)
        end

        selupd_impl!(C, uplo, Ak, Bk, α, 1, tA, cA, tB, cB)
    end

    return C
end

#####
##### rrule implementations
#####

function mul_rrule_impl(A::HermOrSymSparse, B::DenseVecOrMat, ΔC)
    ΔB = A * ΔC
    ΔA = if ΔC isa AbstractZero
        ZeroTangent()
    else
        @thunk begin
            ΔA = similar(A)
            selupd!(ΔA, ΔC, B, 1 / 2, 0)
            ΔA
        end
    end
    return ΔA, ΔB
end

function mul_rrule_impl(A::DenseMat, B::HermSparse, ΔC)
    ΔA = ΔC * B
    ΔB = if ΔC isa AbstractZero
        ZeroTangent()
    else
        @thunk begin
            ΔB = similar(B)
            selupd!(ΔB, A', ΔC', 1 / 2, 0)
            ΔB
        end
    end
    return ΔA, ΔB
end

function mul_rrule_impl(A::DenseMat, B::SymSparse, ΔC)
    ΔA = ΔC * B
    ΔB = if ΔC isa AbstractZero
        ZeroTangent()
    else
        @thunk begin
            ΔB = similar(B)
            selupd!(ΔB, transpose(ΔC), transpose(A), 1 / 2, 0)
            ΔB
        end
    end
    return ΔA, ΔB
end

function dot_rrule_impl(x::StridedVector, A::HermOrSymSparse, y::StridedVector, Ax::StridedVector, Ay::StridedVector, Δz)
    Δx = @thunk Δz * Ay
    Δy = @thunk Δz * Ax

    ΔA = if Δz isa AbstractZero
        ZeroTangent()
    else
        @thunk begin
            ΔA = similar(A)
            selupd!(ΔA, x, y, Δz / 2, 0)
            ΔA
        end
    end

    return Δx, ΔA, Δy
end

#####
##### rrule helpers
#####

function mul_rrule(A::HermOrSymSparse, B::DenseVecOrMat)
    C = A * B

    function pullback(ΔC)
        ΔA, ΔB = mul_rrule_impl(A, B, ΔC)
        return NoTangent(), ΔA, ΔB
    end

    return C, pullback ∘ unthunk
end

function mul_rrule(A::DenseMat, B::HermOrSymSparse)
    C = A * B

    function pullback(ΔC)
        ΔA, ΔB = mul_rrule_impl(A, B, ΔC)
        return NoTangent(), ΔA, ΔB
    end

    return C, pullback ∘ unthunk
end

function dot_rrule(x::StridedVector, A::HermOrSymSparse, y::StridedVector)
    Ax = A * x
    Ay = A * y
    z = dot(x, Ay)

    function pullback(Δz)
        Δx, ΔA, Δy = dot_rrule_impl(x, A, y, Ax, Ay, Δz)
        return NoTangent(), Δx, ΔA, Δy
    end

    return z, pullback ∘ unthunk
end

#####
##### frule implementations
#####

function mul_frule_impl(A, B, dA, dB)
    return A * B, dA * B + A * dB
end

function dot_frule_impl(x::StridedVector, A::HermOrSymSparse, y::StridedVector, dx, dA, dy)
    return dot(x, A, y), dot(dx, A, y) + dot(x, A, dy) + dot(x, dA, y)
end

#####
##### frule / rrule dispatches
#####

for T in (HermSparse, SymSparse)
    # A * X
    @eval function ChainRulesCore.frule((_, dA, dX)::Tuple, ::typeof(*), A::$T, X::DenseVecOrMat)
        return mul_frule_impl(A, X, dA, dX)
    end

    @eval function ChainRulesCore.rrule(::typeof(*), A::$T, X::DenseVecOrMat)
        return mul_rrule(A, X)
    end

    # X * A
    @eval function ChainRulesCore.frule((_, dX, dA)::Tuple, ::typeof(*), X::DenseMat, A::$T)
        return mul_frule_impl(X, A, dX, dA)
    end

    @eval function ChainRulesCore.rrule(::typeof(*), X::DenseMat, A::$T)
        return mul_rrule(X, A)
    end

    # dot(x, A, y) - vectors only, matching upstream ChainRules
    @eval function ChainRulesCore.frule((_, dx, dA, dy)::Tuple, ::typeof(dot), x::StridedVector, A::$T, y::StridedVector)
        return dot_frule_impl(x, A, y, dx, dA, dy)
    end

    @eval function ChainRulesCore.rrule(::typeof(dot), x::StridedVector, A::$T, y::StridedVector)
        return dot_rrule(x, A, y)
    end
end
