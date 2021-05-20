#####
##### `LAPACK.trsyl!`
#####

function ChainRules.frule(
    (_, _, _, ΔA, ΔB, ΔC),
    ::typeof(LAPACK.trsyl!),
    transa::AbstractChar,
    transb::AbstractChar,
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    C::AbstractMatrix{T},
    isgn::Int,
) where {T<:BlasFloat}
    C, scale = LAPACK.trsyl!(transa, transb, A, B, C, isgn)
    Y = (C, scale)
    ΔAtrans = transa === 'T' ? transpose(ΔA) : (transa === 'C' ? ΔA' : ΔA)
    ΔBtrans = transb === 'T' ? transpose(ΔB) : (transb === 'C' ? ΔB' : ΔB)
    mul!(ΔC, ΔAtrans, C, -1, scale)
    mul!(ΔC, C, ΔBtrans, -isgn, true)
    ΔC, scale2 = LAPACK.trsyl!(transa, transb, A, B, ΔC, isgn)
    rmul!(ΔC, inv(scale2))
    ∂Y = Composite{typeof(Y)}(ΔC, Zero())
    return Y, ∂Y
end
