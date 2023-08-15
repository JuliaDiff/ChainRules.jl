function rrule(::typeof(sparse), I::AbstractVector, J::AbstractVector, V::AbstractVector, m, n, combine::typeof(+))
    project_V = ProjectTo(V)
    
    function sparse_pullback(Ω̄)
        ΔΩ = unthunk(Ω̄)
        ΔV = project_V(ΔΩ[I .+ m .* (J .- 1)])
        return NoTangent(), NoTangent(), NoTangent(), ΔV, NoTangent(), NoTangent(), NoTangent()
    end

    return sparse(I, J, V, m, n, combine), sparse_pullback
end

function rrule(::Type{T}, A::AbstractMatrix) where T <: AbstractSparseMatrix
    function sparse_pullback(Ω̄)
        return NoTangent(), Ω̄
    end
    return T(A), sparse_pullback
end

function rrule(::Type{T}, v::AbstractVector) where T <: AbstractSparseVector
    function sparse_pullback(Ω̄)
        return NoTangent(), Ω̄
    end
    return T(v), sparse_pullback
end

function rrule(::typeof(findnz), A::AbstractSparseMatrix)
    I, J, V = findnz(A)
    m, n = size(A)

    function findnz_pullback(Δ)
        _, _, V̄ = unthunk(Δ)
        V̄ isa AbstractZero && return (NoTangent(), V̄)
        return NoTangent(), sparse(I, J, V̄, m, n)
    end

    return (I, J, V), findnz_pullback
end

function rrule(::typeof(findnz), v::AbstractSparseVector)
    I, V = findnz(v)
    n = length(v)

    function findnz_pullback(Δ)
        _, V̄ = unthunk(Δ)
        V̄ isa AbstractZero && return (NoTangent(), V̄)
        return NoTangent(), sparsevec(I, V̄, n)
    end

    return (I, V), findnz_pullback
end

if VERSION < v"1.7"
    #=
    This method for `logabsdet(F::UmfpackLU)` is required to calculate the (log)determinants
    of sparse matrices, but was not defined prior to Julia v1.7. In order fo the rrules
    for the determinants of sparse matrices below to work, they need to be able to 
    compute the primals as well, so this import from the future is included. For more 
    recent versions of Julia, this definition lives in:
    julia/stdlib/SuiteSparse/src/umfpack.jl
    =#
    using SuiteSparse.UMFPACK: _signperm, UmfpackLU

    for itype in (:Int32, :Int64)
        @eval begin
            function LinearAlgebra.logabsdet(F::UmfpackLU{T, $itype}) where {T<:Union{Float64,ComplexF64}} 
                n = checksquare(F)
                issuccess(F) || return log(zero(real(T))), zero(T)
                U = F.U
                Rs = F.Rs
                p = F.p
                q = F.q
                s = _signperm(p)*_signperm(q)*one(real(T))
                P = one(T)
                abs_det = zero(real(T))
                @inbounds for i in 1:n
                    dg_ii = U[i, i] / Rs[i]
                    P *= sign(dg_ii)
                    abs_det += log(abs(dg_ii))
                end
                return abs_det, s * P
            end
        end
    end
end


function rrule(::typeof(logabsdet), x::SparseMatrixCSC)
    F = cholesky(x)
    L, D, U, P = SparseInverseSubset.get_ldup(F)
    Ω = logabsdet(D)
    function logabsdet_pullback(ΔΩ)
        (Δy, Δsigny) = ΔΩ
        (_, signy) = Ω
        f = signy' * Δsigny
        imagf = f - real(f)
        g = real(Δy) + imagf
        Z, P = sparseinv(F, depermute=true)
        ∂x = g * Z'
        return (NoTangent(), ∂x)
    end
    return Ω, logabsdet_pullback
end

function rrule(::typeof(logdet), x::SparseMatrixCSC)
    Ω = logdet(x)
    function logdet_pullback(ΔΩ)
        Z, p = sparseinv(x, depermute=true)
        ∂x = ΔΩ * Z'
        return (NoTangent(), ∂x)
    end
    return Ω, logdet_pullback
end

function rrule(::typeof(det), x::SparseMatrixCSC)
    Ω = det(x)
    function det_pullback(ΔΩ)
        Z, _ = sparseinv(x, depermute=true)
        ∂x = Z' * dot(Ω, ΔΩ)
        return (NoTangent(), ∂x)
    end
    return Ω, det_pullback
end
