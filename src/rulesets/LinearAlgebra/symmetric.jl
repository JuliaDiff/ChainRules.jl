#####
##### `Symmetric`/`Hermitian`
#####

function frule((_, ΔA, _), T::Type{<:LinearAlgebra.HermOrSym}, A::AbstractMatrix, uplo)
    return T(A, uplo), T(ΔA, uplo)
end

function rrule(T::Type{<:LinearAlgebra.HermOrSym}, A::AbstractMatrix, uplo)
    Ω = T(A, uplo)
    function HermOrSym_pullback(ΔΩ)
        return (NO_FIELDS, _symherm_back(T, ΔΩ, Ω.uplo), DoesNotExist())
    end
    return Ω, HermOrSym_pullback
end

function frule((_, ΔA), TM::Type{<:Matrix}, A::LinearAlgebra.HermOrSym)
    return TM(A), TM(_symherm_forward(A, ΔA))
end
function frule((_, ΔA), ::Type{Array}, A::LinearAlgebra.HermOrSym)
    return Array(A), Array(_symherm_forward(A, ΔA))
end

function rrule(TM::Type{<:Matrix}, A::LinearAlgebra.HermOrSym)
    function Matrix_pullback(ΔΩ)
        TA = _symhermtype(A)
        T∂A = TA{eltype(ΔΩ),typeof(ΔΩ)}
        uplo = A.uplo
        ∂A = T∂A(_symherm_back(A, ΔΩ, uplo), uplo)
        return NO_FIELDS, ∂A
    end
    return TM(A), Matrix_pullback
end
rrule(::Type{Array}, A::LinearAlgebra.HermOrSym) = rrule(Matrix, A)

# Get type (Symmetric or Hermitian) from type or matrix
_symhermtype(::Type{<:Symmetric}) = Symmetric
_symhermtype(::Type{<:Hermitian}) = Hermitian
_symhermtype(A) = _symhermtype(typeof(A))

# for Ω = Matrix(A::HermOrSym), push forward ΔA to get ∂Ω
function _symherm_forward(A, ΔA)
    TA = _symhermtype(A)
    return if ΔA isa TA
        ΔA
    else
        TA{eltype(ΔA),typeof(ΔA)}(ΔA, A.uplo)
    end
end

# for Ω = HermOrSym(A, uplo), pull back ΔΩ to get ∂A
_symherm_back(::Type{<:Symmetric}, ΔΩ, uplo) = _symmetric_back(ΔΩ, uplo)
function _symherm_back(::Type{<:Hermitian}, ΔΩ::AbstractMatrix{<:Real}, uplo)
    return _symmetric_back(ΔΩ, uplo)
end
_symherm_back(::Type{<:Hermitian}, ΔΩ, uplo) = _hermitian_back(ΔΩ, uplo)
_symherm_back(Ω, ΔΩ, uplo) = _symherm_back(typeof(Ω), ΔΩ, uplo)

function _symmetric_back(ΔΩ, uplo)
    L, U, D = LowerTriangular(ΔΩ), UpperTriangular(ΔΩ), Diagonal(ΔΩ)
    return uplo == 'U' ? U .+ transpose(L) - D : L .+ transpose(U) - D
end
_symmetric_back(ΔΩ::Diagonal, uplo) = ΔΩ
_symmetric_back(ΔΩ::UpperTriangular, uplo) = Matrix(uplo == 'U' ? ΔΩ : transpose(ΔΩ))
_symmetric_back(ΔΩ::LowerTriangular, uplo) = Matrix(uplo == 'U' ? transpose(ΔΩ) : ΔΩ)

function _hermitian_back(ΔΩ, uplo)
    L, U, rD = LowerTriangular(ΔΩ), UpperTriangular(ΔΩ), real.(Diagonal(ΔΩ))
    return uplo == 'U' ? U .+ L' - rD : L .+ U' - rD
end
_hermitian_back(ΔΩ::Diagonal, uplo) = real.(ΔΩ)
function _hermitian_back(ΔΩ::LinearAlgebra.AbstractTriangular, uplo)
    ∂UL = ΔΩ .- Diagonal(_extract_imag.(diag(ΔΩ)))
    return if istriu(ΔΩ)
        return Matrix(uplo == 'U' ? ∂UL : ∂UL')
    else
        return Matrix(uplo == 'U' ? ∂UL' : ∂UL)
    end
end

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
            return _hermitrizeback!(U * U′∂AU * U', A)
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
function _hermitrizeback!(∂A, A)
    @inbounds for i in axes(∂A, 1)
        for j in 1:(i - 1)
            if A.uplo === 'U'
                ∂A[j, i] = (∂A[j, i] + ∂A[i, j]') / 2
            else
                ∂A[i, j] = (∂A[i, j] + ∂A[j, i]') / 2
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
            return _hermitrizeback!(U * U′∂AU * U', A)
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
                    return _hermitrizeback!(U * U′∂AU * U', A)
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
            return _hermitrizeback!(U * U′∂AU * U', A)
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
