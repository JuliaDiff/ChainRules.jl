# matrix functions of dense matrices

# NOTE: for matrix functions whose power series representation has real coefficients,
# the pullback and pushforward are related by an adjoint.
# Specifically, if the pushforward of f(A) is (f_*)_A(ΔA), then the pullback at Y=f(A) is
# (f^*)_Y(ΔY) = (f_*)_{A'}(ΔY) = ((f_*)_A(ΔY'))'
# So we reuse the code from the pushforward to implement the pullback.

"""
    _matfun(f, A) -> (Y, intermediates)

Compute the matrix function `Y=f(A)` for matrix `A`.
The function returns a tuple containing the result and a tuple of intermediates to be
reused by `_matfun_frechet` to compute the Fréchet derivative.
Note that any function `f` used with this **must** have a `frule` defined on it.
"""
_matfun

"""
    _matfun!(f, A) -> (Y, intermediates)

Similar to [`_matfun`](@ref), but where `A` may be overwritten.
"""
_matfun!

"""
    _matfun_frechet(f, A, Y, ΔA, intermediates)

Compute the Fréchet derivative of the matrix function `Y=f(A)`, where the Fréchet derivative
of `A` is `ΔA`, and `intermediates` is the second argument returned by `_matfun`.
"""
_matfun_frechet

"""
    _matfun_frechet!(f, A, Y, ΔA, intermediates)

Similar to `_matfun_frechet!`, but where `ΔA` may be overwritten.
"""
_matfun_frechet!

function frule((_, ΔA), ::typeof(LinearAlgebra.exp!), A::StridedMatrix{<:BlasFloat})
    if ishermitian(A)
        hermX, ∂hermX = frule((Zero(), ΔA), exp, Hermitian(A))
        X = LinearAlgebra.copytri!(parent(hermX), 'U', true)
        if ∂hermX isa LinearAlgebra.RealHermSymComplexHerm
            ∂X = LinearAlgebra.copytri!(parent(∂hermX), 'U', true)
        else
            ∂X = ∂hermX
        end
    else
        X, intermediates = _matfun!(exp, A)
        ∂X = _matfun_frechet!(exp, A, X, ΔA, intermediates)
    end
    return X, ∂X
end

function rrule(::typeof(exp), A0::StridedMatrix{<:BlasFloat})
    # TODO: try to make this more type-stable
    if ishermitian(A0)
        # call _matfun instead of the rrule to avoid hermitrizing ∂A in the pullback
        hermA = Hermitian(A0)
        hermX, hermX_intermediates = _matfun(exp, hermA)
        function exp_pullback_hermitian(ΔX)
            ∂hermA = _matfun_frechet(exp, hermA, hermX, ΔX, hermX_intermediates)
            ∂hermA isa LinearAlgebra.RealHermSymComplexHerm || return NO_FIELDS, ∂hermA
            return NO_FIELDS, parent(∂hermA)
        end
        return LinearAlgebra.copytri!(parent(hermX), 'U', true), exp_pullback_hermitian
    else
        A = copy(A0)
        X, intermediates = _matfun!(exp, A)
        function exp_pullback(ΔX)
            ΔX′ = copy(adjoint(ΔX))
            ∂A′ = _matfun_frechet!(exp, A, X, ΔX′, intermediates)
            ∂A = copy(adjoint(∂A′))
            return NO_FIELDS, ∂A
        end
        return X, exp_pullback
    end
end

## Destructive matrix exponential using algorithm from Higham, 2008,
## "Functions of Matrices: Theory and Computation", SIAM
## Adapted from LinearAlgebra.exp! with return of intermediates
function _matfun!(::typeof(exp), A::StridedMatrix{T}) where T<:BlasFloat
    n = LinearAlgebra.checksquare(A)
    ilo, ihi, scale = LAPACK.gebal!('B', A)    # modifies A
    nA   = opnorm(A, 1)
    Inn    = Matrix{T}(I, n, n)
    ## For sufficiently small nA, use lower order Padé-Approximations
    if (nA <= 2.1)
        if nA > 0.95
            C = T[17643225600.,8821612800.,2075673600.,302702400.,
                     30270240.,   2162160.,    110880.,     3960.,
                           90.,         1.]
        elseif nA > 0.25
            C = T[17297280.,8648640.,1995840.,277200.,
                     25200.,   1512.,     56.,     1.]
        elseif nA > 0.015
            C = T[30240.,15120.,3360.,
                    420.,   30.,   1.]
        else
            C = T[120.,60.,12.,1.]
        end
        si = 0
    else
        C = T[64764752532480000.,32382376266240000.,7771770303897600.,
                1187353796428800.,  129060195264000.,  10559470521600.,
                    670442572800.,      33522128640.,      1323241920.,
                        40840800.,           960960.,           16380.,
                             182.,                1.]
        s = log2(nA/5.4)               # power of 2 later reversed by squaring
        si = ceil(Int,s)
    end

    if si > 0
        A ./= convert(T,2^si)
    end

    A2 = A * A
    P  = copy(Inn)
    W  = C[2] * P
    V  = C[1] * P
    Apows = typeof(P)[]
    for k in 1:(div(size(C, 1), 2) - 1)
        k2 = 2 * k
        P *= A2
        push!(Apows, P)
        W += C[k2 + 2] * P
        V += C[k2 + 1] * P
    end
    U = A * W
    X = V + U
    F = lu!(V-U) # NOTE: use lu! instead of LAPACK.gesv! so we can reuse factorization
    ldiv!(F, X)
    Xpows = typeof(X)[X]
    if si > 0            # squaring to reverse dividing by power of 2
        for t=1:si
            X *= X
            push!(Xpows, X)
        end
    end

    # Undo the balancing
    for j = ilo:ihi
        scj = scale[j]
        for i = 1:n
            X[j,i] *= scj
        end
        for i = 1:n
            X[i,j] /= scj
        end
    end

    if ilo > 1       # apply lower permutations in reverse order
        for j in (ilo-1):-1:1; LinearAlgebra.rcswap!(j, Int(scale[j]), X) end
    end
    if ihi < n       # apply upper permutations in forward order
        for j in (ihi+1):n;    LinearAlgebra.rcswap!(j, Int(scale[j]), X) end
    end
    return X, (ilo, ihi, scale, C, si, Apows, W, F, Xpows)
end

function _matfun_frechet!(
    ::typeof(exp),
    A::StridedMatrix{T},
    X,
    ΔA,
    (ilo, ihi, scale, C, si, Apows, W, F, Xpows),
) where {T<:BlasFloat}
    n = LinearAlgebra.checksquare(A)
    for j = ilo:ihi
        scj = scale[j]
        for i = 1:n
            ΔA[j,i] /= scj
        end
        for i = 1:n
            ΔA[i,j] *= scj
        end
    end

    if si > 0
        ΔA ./= convert(T, 2^si)
    end

    ∂A2 = mul!(A * ΔA, ΔA, A, true, true)
    A2 = first(Apows)
    # we will repeatedly overwrite ∂temp and ∂P below
    ∂temp = Matrix{eltype(∂A2)}(undef, n, n)
    ∂P = copy(∂A2)
    ∂W = C[4] * ∂P
    ∂V = C[3] * ∂P
    for k in 2:(length(Apows)-1)
        k2 = 2 * k
        P = Apows[k - 1]
        ∂P, ∂temp = mul!(mul!(∂temp, ∂P, A2), P, ∂A2, true, true), ∂P
        axpy!(C[k2 + 2], ∂P, ∂W)
        axpy!(C[k2 + 1], ∂P, ∂V)
    end
    ∂U, ∂temp = mul!(mul!(∂temp, A, ∂W), ΔA, W, true, true), ∂W
    ∂temp .= ∂U .- ∂V
    ∂X = add!!(∂U, ∂V)
    mul!(∂X, ∂temp, first(Xpows), true, true)
    ldiv!(F, ∂X)

    if si > 0
        for t = 1:(length(Xpows)-1)
            X = Xpows[t]
            ∂X, ∂temp = mul!(mul!(∂temp, X, ∂X), ∂X, X, true, true), ∂X
        end
    end

    for j = ilo:ihi
        scj = scale[j]
        for i = 1:n
            ∂X[j,i] *= scj
        end
        for i = 1:n
            ∂X[i,j] /= scj
        end
    end

    if ilo > 1       # apply lower permutations in reverse order
        for j in (ilo-1):-1:1
            LinearAlgebra.rcswap!(j, Int(scale[j]), ∂X)
        end
    end
    if ihi < n       # apply upper permutations in forward order
        for j in (ihi+1):n
            LinearAlgebra.rcswap!(j, Int(scale[j]), ∂X)
        end
    end
    return ∂X
end
