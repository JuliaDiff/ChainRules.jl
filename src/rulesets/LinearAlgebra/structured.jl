# Structured matrices


#####
##### `Diagonal`
#####

function rrule(::Type{<:Diagonal}, d::AbstractVector)
    function Diagonal_pullback(ȳ::AbstractMatrix)
        return (NO_FIELDS, diag(ȳ))
    end
    function Diagonal_pullback(ȳ::Composite)
        # TODO: Assert about the primal type in the Composite, It should be Diagonal
        # infact it should be exactly the type of `Diagonal(d)`
        # but right now Zygote loses primal type information so we can't use it.
        # See https://github.com/FluxML/Zygote.jl/issues/603
        return (NO_FIELDS, ȳ.diag)
    end
    return Diagonal(d), Diagonal_pullback
end

function rrule(::typeof(diag), A::AbstractMatrix)
    function diag_pullback(ȳ)
        return (NO_FIELDS, @thunk(Diagonal(ȳ)))
    end
    return diag(A), diag_pullback
end
if VERSION ≥ v"1.3"
    function rrule(::typeof(diag), A::AbstractMatrix, k::Integer)
        function diag_pullback(ȳ)
            return (NO_FIELDS, @thunk(diagm(size(A)..., k => ȳ)), DoesNotExist())
        end
        return diag(A, k), diag_pullback
    end

    function rrule(::typeof(diagm), m::Integer, n::Integer, kv::Pair{<:Integer,<:AbstractVector}...)
        function diagm_pullback(ȳ)
            return (NO_FIELDS, DoesNotExist(), DoesNotExist(), _diagm_back.(kv, Ref(ȳ))...)
        end
        return diagm(m, n, kv...), diagm_pullback
    end
end
function rrule(::typeof(diagm), kv::Pair{<:Integer,<:AbstractVector}...)
    function diagm_pullback(ȳ)
        return (NO_FIELDS, _diagm_back.(kv, Ref(ȳ))...)
    end
    return diagm(kv...), diagm_pullback
end

function _diagm_back(p, ȳ)
    return Thunk() do
        k, v = p
        d = diag(ȳ, k)[1:length(v)] # handle if diagonal was smaller than matrix
        return Composite{typeof(p)}(second = d)
    end
end

function rrule(::typeof(*), D::Diagonal{<:Real}, V::AbstractVector{<:Real})
    function times_pullback(Ȳ)
        return (NO_FIELDS, @thunk(Diagonal(Ȳ .* V)), @thunk(D * Ȳ))
    end
    return D * V, times_pullback
end

#####
##### `Symmetric`
#####

function rrule(::Type{<:Symmetric}, A::AbstractMatrix)
    function Symmetric_pullback(ȳ)
        return (NO_FIELDS, @thunk(_symmetric_back(ȳ)))
    end
    return Symmetric(A), Symmetric_pullback
end

_symmetric_back(ΔΩ) = UpperTriangular(ΔΩ) + transpose(LowerTriangular(ΔΩ)) - Diagonal(ΔΩ)
_symmetric_back(ΔΩ::Union{Diagonal,UpperTriangular}) = ΔΩ

#####
##### `Symmetric{<:Real}`/`Hermitian` eigendecomposition
#####

function frule((_, ΔA), ::typeof(eigen), A::LinearAlgebra.RealHermSymComplexHerm)
    F = eigen(A)
    λ, U = F
    ∂Λ = U' * ΔA * U
    ∂λ = real(diag(∂Λ)) # if ΔA is Hermitian, so is ∂Λ, so its diagonal is real
    # K is skew-hermitian with zero diag
    K = ∂Λ ./ _nonzero.(λ' .- λ)
    _setdiag!(K, Zero())
    ∂U = U * K
    ∂F = Composite{typeof(F)}(values = ∂λ, vectors = ∂U)
    return F, ∂F
end

function rrule(::typeof(eigen), A::LinearAlgebra.RealHermSymComplexHerm)
    F = eigen(A)
    function eigen_pullback(ΔF)
        ∂A = Thunk() do
            λ, U = F
            ∂λ, ∂U = ΔF.values, ΔF.vectors
            if ∂U isa AbstractZero
                U′∂AU = Diagonal(∂λ)
            else
                K = U' * ∂U
                # unstable for degenerate matrices
                U′∂AU = K ./ _nonzero.(λ' .- λ)
                _setdiag!(U′∂AU, ∂λ)
            end
            return _symhermback!(U * U′∂AU * U', A)
        end
        return NO_FIELDS, ∂A
    end
    return F, eigen_pullback
end

function frule((_, ΔA), ::typeof(eigvals), A::LinearAlgebra.RealHermSymComplexHerm)
    λ, U = eigen(A)
    ∂Λ = U' * ΔA * U
    ∂λ = real(diag(∂Λ)) # if ΔA is Hermitian, so is ∂Λ, so its diagonal is real
    return λ, ∂λ
end

function rrule(::typeof(eigvals), A::LinearAlgebra.RealHermSymComplexHerm)
    F, back = rrule(eigen, A)
    λ, U = F
    function eigvals_pullback(Δλ)
        ∂A = Thunk() do
            ∂F = Composite{typeof(F)}(values = Δλ)
            _, ∂A = back(∂F)
            return unthunk(∂A)
        end
        return NO_FIELDS, ∂A
    end
    return λ, eigvals_pullback
end

# if |x| < eps(), return a small number with its sign, where zero has a positive sign.
_nonzero(x) = ifelse(signbit(x), min(x, -eps(eltype(x))), max(x, eps(eltype(x))))

function _setdiag!(A, d)
    for i in axes(A, 1)
        @inbounds A[i, i] = d isa AbstractZero ? 0 : d[i]
    end
    return A
end

_pureimag(x) = x - real(x)

function _realifydiag!(A)
    for i in axes(A, 1)
        @inbounds A[i, i] = real(A[i, i])
    end
    return A
end
_realifydiag!(A::AbstractMatrix{<:Real}) = A

_realifydiag(A) = A .- Diagonal(_pureimag.(diag(A)))
_realifydiag(A::AbstractMatrix{<:Real}) = A

_symherm(A::AbstractMatrix{<:Real}, uplo = :U) = Symmetric(A, uplo)
_symherm(A::AbstractMatrix{<:Complex}, uplo = :U) = Hermitian(A, uplo)

_symhermtype(A::Symmetric) = Symmetric
_symhermtype(A::Hermitian) = Hermitian

function _symhermlike!(A, S::LinearAlgebra.RealHermSymComplexHerm)
    _realifydiag!(A)
    return typeof(S)(A, S.uplo)
end

function _symhermfwd!(A, uplo = :U)
    _realifydiag!(A)
    return _symherm(A, uplo)
end

# pullback of hermitrization
function _symhermback!(∂A, A)
    @inbounds for i in axes(∂A, 1)
        for j in 1:(i - 1)
            if A.uplo === 'U'
                ∂A[j, i] += ∂A[i, j]
                ∂A[i, j] = 0
            else
                ∂A[i, j] += ∂A[j, i]
                ∂A[j, i] = 0
            end
        end
        if eltype(∂A) <: Complex
            ∂A[i, i] = real(∂A[i, i])
        end
    end
    return typeof(A)(∂A, A.uplo)
end

#####
##### `Symmetric{<:Real}`/`Hermitian` power series functions
#####

# Currently only defined for series functions whose codomain is ℝ
# These are type-stable and closed under `func`

# The efficient way to do this is probably to AD Base.power_by_squaring
function frule((_, ΔA, _), ::typeof(^), A::LinearAlgebra.RealHermSymComplexHerm, p::Integer)
    λ, U = eigen(A)
    λᵖ = λ .^ p
    Y = U * Diagonal(λᵖ) * U'
    _realifydiag!(Y)
    Y = _symhermtype(A)(Y, :U)
    dλᵖ_dλ = p .* λ .^ (p - 1)
    ∂Λ = U' * ΔA * U
    U′∂YU = _muldiffquotmat(λ, λᵖ, dλᵖ_dλ, ∂Λ)
    ∂Y = _symhermlike!(U * U′∂YU * U', Y)
    return Y, ∂Y
end

function rrule(::typeof(^), A::LinearAlgebra.RealHermSymComplexHerm, p::Integer)
    λ, U = eigen(A)
    λᵖ = λ .^ p
    Y = U * Diagonal(λᵖ) * U'
    _realifydiag!(Y)
    Y = _symhermtype(A)(Y, :U)
    function pow_pullback(ΔY)
        ∂A = Thunk() do
            dλᵖ_dλ = p .* λ .^ (p - 1)
            ∂Λᵖ = U' * _realifydiag(ΔY) * U
            U′∂AU = _muldiffquotmat(λ, λᵖ, dλᵖ_dλ, ∂Λᵖ)
            return _symhermback!(U * U′∂AU * U', A)
        end
        return NO_FIELDS, ∂A, DoesNotExist()
    end
    return Y, pow_pullback
end

# TODO: support log, sqrt, acos, asin, and non-int pow, which are type-unstable
for func in (:exp, :cos, :sin, :tan, :cosh, :sinh, :tanh, :atan, :asinh, :atanh)
    @eval begin
        function frule((_, ΔA), ::typeof($func), A::LinearAlgebra.RealHermSymComplexHerm)
            df = λi -> frule((Zero(), One()), $func, λi)
            λ, U = eigen(A)
            fλ_df_dλ = df.(λ)
            fλ = first.(fλ_df_dλ)
            Y = _symhermfwd!(U * Diagonal(fλ) * U')
            df_dλ = last.(unthunk.(fλ_df_dλ))
            ∂Λ = U' * ΔA * U
            U′∂YU = _muldiffquotmat(λ, fλ, df_dλ, ∂Λ)
            ∂Y = _symhermlike!(U * U′∂YU * U', Y)
            return Y, ∂Y
        end

        function rrule(::typeof($func), A::LinearAlgebra.RealHermSymComplexHerm)
            df = λi -> frule((Zero(), One()), $func, λi)
            λ, U = eigen(A)
            fλ_df_dλ = df.(λ)
            fλ = first.(fλ_df_dλ)
            Y = _symhermfwd!(U * Diagonal(fλ) * U')
            function $(Symbol("$(func)_pullback"))(ΔY)
                ∂A = Thunk() do
                    df_dλ = unthunk.(last.(fλ_df_dλ))
                    ∂fΛ = U' * _realifydiag(ΔY) * U
                    U′∂AU = _muldiffquotmat(λ, fλ, df_dλ, ∂fΛ)
                    return _symhermback!(U * U′∂AU * U', A)
                end
                return NO_FIELDS, ∂A
            end
            return Y, $(Symbol("$(func)_pullback"))
        end
    end
end

function frule((_, ΔA), ::typeof(sincos), A::LinearAlgebra.RealHermSymComplexHerm)
    λ, U = eigen(A)
    sinλ, cosλ = sin.(λ), cos.(λ)
    sinA = _symhermfwd!(U * Diagonal(sinλ) * U')
    cosA = _symhermfwd!(U * Diagonal(cosλ) * U')
    sincosA = (sinA, cosA)
    ∂Λ = U' * ΔA * U
    U′∂sinAU = _muldiffquotmat(λ, sinλ, cosλ, ∂Λ)
    ∂sinA = _symhermlike!(U * U′∂sinAU * U', sinA)
    U′∂cosAU = _muldiffquotmat(λ, cosλ, -sinλ, ∂Λ)
    ∂cosA = _symhermlike!(U * U′∂cosAU * U', cosA)
    ∂sincosA = Composite{typeof(sincosA)}(∂sinA, ∂cosA)
    return sincosA, ∂sincosA
end

function rrule(::typeof(sincos), A::LinearAlgebra.RealHermSymComplexHerm)
    λ, U = eigen(A)
    sinλ, cosλ = sin.(λ), cos.(λ)
    sinA = _symhermfwd!(U * Diagonal(sinλ) * U')
    cosA = _symhermfwd!(U * Diagonal(cosλ) * U')
    sincosA = (sinA, cosA)
    function sincos_pullback(ΔsincosA)
        ∂A = Thunk() do
            ΔsinA, ΔcosA = ΔsincosA
            ∂sinΛ, ∂cosΛ = U' * _realifydiag(ΔsinA) * U, U' * _realifydiag(ΔcosA) * U
            inds = eachindex(λ)
            U′∂AU = @inbounds begin
                _diffquot.(inds, inds', Ref(λ), Ref(sinλ), Ref(cosλ)) .* ∂sinΛ .+
                _diffquot.(inds, inds', Ref(λ), Ref(cosλ), Ref(-sinλ)) .* ∂cosΛ
            end
            return _symhermback!(U * U′∂AU * U', A)
        end
        return NO_FIELDS, ∂A
    end
    return sincosA, sincos_pullback
end

# difference quotient, i.e. Pᵢⱼ = (f(λᵢ) - f(λⱼ)) / (λᵢ - λⱼ), with f'(λᵢ) when i==j
Base.@propagate_inbounds function _diffquot(i, j, λ, fλ, df_dλ)
    T = typeof(zero(eltype(fλ)) / one(eltype(λ)) + zero(eltype(df_dλ)))
    i == j && return T(df_dλ[i])
    Δλ = λ[i] - λ[j]
    # handle round-off error by taylor expanding Δfλ / Δλ as Δλ → 0
    # total error on the order of eps()^(2/3)
    abs(Δλ) < cbrt(eps(real(T))) && return T((df_dλ[i] + df_dλ[j]) / 2)
    return T((fλ[i] - fλ[j]) / Δλ)
end

# multiply Δ by the matrix of difference quotients P
function _muldiffquotmat(λ, fλ, df_dλ, Δ)
    inds = eachindex(λ)
    return @inbounds _diffquot.(inds, inds', Ref(λ), Ref(fλ), Ref(df_dλ)) .* Δ
end

#####
##### `Adjoint`
#####

# ✖️✖️✖️TODO: Deal with complex-valued arrays as well
function rrule(::Type{<:Adjoint}, A::AbstractMatrix{<:Real})
    function Adjoint_pullback(ȳ)
        return (NO_FIELDS, @thunk(adjoint(ȳ)))
    end
    return Adjoint(A), Adjoint_pullback
end

function rrule(::Type{<:Adjoint}, A::AbstractVector{<:Real})
    function Adjoint_pullback(ȳ)
        return (NO_FIELDS, @thunk(vec(adjoint(ȳ))))
    end
    return Adjoint(A), Adjoint_pullback
end

function rrule(::typeof(adjoint), A::AbstractMatrix{<:Real})
    function adjoint_pullback(ȳ)
        return (NO_FIELDS, @thunk(adjoint(ȳ)))
    end
    return adjoint(A), adjoint_pullback
end

function rrule(::typeof(adjoint), A::AbstractVector{<:Real})
    function adjoint_pullback(ȳ)
        return (NO_FIELDS, @thunk(vec(adjoint(ȳ))))
    end
    return adjoint(A), adjoint_pullback
end

#####
##### `Transpose`
#####

function rrule(::Type{<:Transpose}, A::AbstractMatrix)
    function Transpose_pullback(ȳ)
        return (NO_FIELDS, @thunk transpose(ȳ))
    end
    return Transpose(A), Transpose_pullback
end

function rrule(::Type{<:Transpose}, A::AbstractVector)
    function Transpose_pullback(ȳ)
        return (NO_FIELDS, @thunk vec(transpose(ȳ)))
    end
    return Transpose(A), Transpose_pullback
end

function rrule(::typeof(transpose), A::AbstractMatrix)
    function transpose_pullback(ȳ)
        return (NO_FIELDS, @thunk transpose(ȳ))
    end
    return transpose(A), transpose_pullback
end

function rrule(::typeof(transpose), A::AbstractVector)
    function transpose_pullback(ȳ)
        return (NO_FIELDS, @thunk vec(transpose(ȳ)))
    end
    return transpose(A), transpose_pullback
end

#####
##### Triangular matrices
#####

function rrule(::Type{<:UpperTriangular}, A::AbstractMatrix)
    function UpperTriangular_pullback(ȳ)
        return (NO_FIELDS, @thunk Matrix(ȳ))
    end
    return UpperTriangular(A), UpperTriangular_pullback
end

function rrule(::Type{<:LowerTriangular}, A::AbstractMatrix)
    function LowerTriangular_pullback(ȳ)
        return (NO_FIELDS, @thunk Matrix(ȳ))
    end
    return LowerTriangular(A), LowerTriangular_pullback
end

function rrule(::typeof(triu), A::AbstractMatrix, k::Integer)
    function triu_pullback(ȳ)
        return (NO_FIELDS, @thunk(triu(ȳ, k)), DoesNotExist())
    end
    return triu(A, k), triu_pullback
end
function rrule(::typeof(triu), A::AbstractMatrix)
    function triu_pullback(ȳ)
        return (NO_FIELDS, @thunk triu(ȳ))
    end
    return triu(A), triu_pullback
end

function rrule(::typeof(tril), A::AbstractMatrix, k::Integer)
    function tril_pullback(ȳ)
        return (NO_FIELDS, @thunk(tril(ȳ, k)), DoesNotExist())
    end
    return tril(A, k), tril_pullback
end
function rrule(::typeof(tril), A::AbstractMatrix)
    function tril_pullback(ȳ)
        return (NO_FIELDS, @thunk tril(ȳ))
    end
    return tril(A), tril_pullback
end
