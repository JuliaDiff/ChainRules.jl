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

_symmetric_back(ΔΩ) = UpperTriangular(ΔΩ) + LowerTriangular(ΔΩ)' - Diagonal(ΔΩ)
_symmetric_back(ΔΩ::Union{Diagonal,UpperTriangular}) = ΔΩ

#####
##### `Symmetric{<:Real}`/`Hermitian` eigendecomposition
#####

function frule((_, ΔA), ::typeof(eigen), A::LinearAlgebra.RealHermSymComplexHerm)
    F = eigen(A)
    ∂F = Thunk() do
        λ, U = F
        ∂Λ = U' * ΔA * U
        ∂λ = real(diag(∂Λ)) # if ΔA is Hermitian, so is ∂Λ, so its diagonal is real
        # K is skew-hermitian with zero diag
        K = ∂Λ ./ _nonzero.(λ' .- λ)
        _setdiag!(K, Zero())
        ∂U = U * K
        return Composite{typeof(F)}(values = ∂λ, vectors = ∂U)
    end
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
                # K is skew-hermitian
                K = U' * ∂U
                # unstable for degenerate matrices
                U′∂AU = K ./ _nonzero.(λ' .- λ)
                _setdiag!(U′∂AU, ∂λ)
            end
            return _symhermlike(U * U′∂AU * U', A)
        end
        return NO_FIELDS, ∂A
    end
    return F, eigen_pullback
end

function frule((_, ΔA), ::typeof(eigvals), A::LinearAlgebra.RealHermSymComplexHerm)
    λ, U = eigen(A)
    return λ, @thunk real(diag(U' * ΔA * U))
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

_setdiag!(A, d) = (A[diagind(A)] = d)
_setdiag!(A, d::AbstractZero) = (A[diagind(A)] .= 0)

_pureimag(x) = x - real(x)

function _realifydiag!(A)
    for i in diagind(A)
        @inbounds A[i] = real(A[i])
    end
    return A
end
_realifydiag!(A::LinearAlgebra.RealHermSym) = A

_realifydiag(A) = A - _pureimag(Diagonal(A))
_realifydiag(A::Union{Hermitian{<:Real},Symmetric{<:Real}}) = A

_symherm(A::AbstractMatrix{<:Real}, uplo = :U) = Symmetric(A, uplo)
_symherm(A::AbstractMatrix{<:Complex}, uplo = :U) = Hermitian(A, uplo)

# constrain B to have same Symmetric/Hermitian type as A
_symhermlike(A, S::Symmetric, uplo = Symbol(S.uplo)) = Symmetric(A, uplo)
_symhermlike(A, S::Hermitian, uplo = Symbol(S.uplo)) = Hermitian(A, uplo)

# call _symherm but enforce the real diagonal constraint on the `data` field.
function _symhermdata!(args...)
    S = _symherm(args...)
    _realifydiag!(S)
    return S
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
    Y = _symhermdata!(U * Diagonal(λᵖ) * U')
    ∂Y = Thunk() do
        dλᵖ_dλ = p .* λ .^ (p - 1)
        ∂Λ = U' * ΔA * U
        U′∂YU = _muldiffquotmat(λ, λᵖ, dλᵖ_dλ, ∂Λ)
        return _symhermdata!(U * U′∂YU * U', Y)
    end
    return Y, ∂Y
end

function rrule(::typeof(^), A::LinearAlgebra.RealHermSymComplexHerm, p::Integer)
    λ, U = eigen(A)
    λᵖ = λ .^ p
    Y = _symhermdata!(U * Diagonal(λᵖ) * U')
    function pow_pullback(ΔY)
        ∂A = Thunk() do
            dλᵖ_dλ = p .* λ .^ (p - 1)
            ∂Λᵖ = U' * _realifydiag(ΔY) * U
            # TODO: make sure that the `conj` is needed
            U′∂AU = _muldiffquotmat(λ, λᵖ, dλᵖ_dλ, ∂Λᵖ)
            return _symhermlike(U * U′∂AU * U', A)
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
            Y = _symhermdata!(U * Diagonal(fλ) * U')
            ∂Y = Thunk() do
                df_dλ = last.(unthunk.(fλ_df_dλ))
                ∂Λ = U' * ΔA * U
                U′∂YU = _muldiffquotmat(λ, fλ, df_dλ, ∂Λ)
                return _symhermdata!(U * U′∂YU * U', Y)
            end
            return Y, ∂Y
        end

        function rrule(::typeof($func), A::LinearAlgebra.RealHermSymComplexHerm)
            df = λi -> frule((Zero(), One()), $func, λi)
            λ, U = eigen(A)
            fλ_df_dλ = df.(λ)
            fλ = first.(fλ_df_dλ)
            Y = _symhermdata!(U * Diagonal(fλ) * U')
            function $(Symbol("$(func)_pullback"))(ΔY)
                ∂A = Thunk() do
                    df_dλ = unthunk.(last.(fλ_df_dλ))
                    ∂fΛ = U' * _realifydiag(ΔY) * U
                    U′∂AU = _muldiffquotmat(λ, fλ, df_dλ, ∂fΛ)
                    return _symhermlike(U * U′∂AU * U', A)
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
    sinA = _symhermdata!(U * Diagonal(sinλ) * U')
    cosA = _symhermdata!(U * Diagonal(cosλ) * U')
    sincosA = (sinA, cosA)
    ∂sincosA = Thunk() do
        ∂Λ = U' * ΔA * U
        U′∂sinAU = _muldiffquotmat(λ, sinλ, cosλ, ∂Λ)
        ∂sinA = _symhermdata!(U * U′∂sinAU * U', sinA)
        U′∂cosAU = _muldiffquotmat(λ, cosλ, -sinλ, ∂Λ)
        ∂cosA = _symhermdata!(U * U′∂cosAU * U', cosA)
        return Composite{typeof(sincosA)}(∂sinA, ∂cosA)
    end
    return sincosA, ∂sincosA
end

function rrule(::typeof(sincos), A::LinearAlgebra.RealHermSymComplexHerm)
    λ, U = eigen(A)
    sinλ, cosλ = sin.(λ), cos.(λ)
    sinA = _symhermdata!(U * Diagonal(sinλ) * U')
    cosA = _symhermdata!(U * Diagonal(cosλ) * U')
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
            return _symhermlike(U * U′∂AU * U', A)
        end
        return NO_FIELDS, ∂A
    end
    return sincosA, sincos_pullback
end

# difference quotient, i.e. Pᵢⱼ = (f(λᵢ) - f(λⱼ)) / (λᵢ - λⱼ), with f'(λᵢ) when i==j
Base.@propagate_inbounds function _diffquot(i, j, λ, fλ, df_dλ)
    i == j && return df_dλ[i]
    Δλ = λ[i] - λ[j]
    T = real(eltype(λ))
    # Handle degenerate eigenvalues by taylor expanding Δfλ / Δλ as Δλ → 0
    abs2(Δλ) < eps(T) && return (df_dλ[i] + df_dλ[j]) / 2
    Δfλ = fλ[i] - fλ[j]
    return Δfλ / Δλ
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
