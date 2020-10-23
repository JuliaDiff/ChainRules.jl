#####
##### `dot`
#####

function frule((_, Δx, Δy), ::typeof(dot), x, y)
    return dot(x, y), dot(Δx, y) + dot(x, Δy)
end

function rrule(::typeof(dot), x, y)
    function dot_pullback(ΔΩ)
        return (NO_FIELDS, @thunk(y .* ΔΩ'), @thunk(x .* ΔΩ))
    end
    return dot(x, y), dot_pullback
end

function frule((_, Δx, ΔA, Δy), ::typeof(dot), x::AbstractVector{<:Number}, A::AbstractMatrix{<:Number}, y::AbstractVector{<:Number})
    return dot(x, A, y), dot(Δx, A, y) + dot(x, ΔA, y) + dot(x, A, Δy)
end

function rrule(::typeof(dot), x::AbstractVector{<:Number}, A::AbstractMatrix{<:Number}, y::AbstractVector{<:Number})
    Ay = A * y
    z = adjoint(x) * Ay
    function dot_pullback(ΔΩ)
        dx = @thunk conj(ΔΩ) .* Ay
        dA = @thunk ΔΩ .* x .* adjoint(y)
        dy = @thunk ΔΩ .* (adjoint(A) * x)
        return (NO_FIELDS, dx, dA, dy)
    end
    dot_pullback(::Zero) = (NO_FIELDS, Zero(), Zero(), Zero())
    return z, dot_pullback
end

function rrule(::typeof(dot), x::AbstractVector{<:Number}, A::Diagonal{<:Number}, y::AbstractVector{<:Number})
    z = dot(x,A,y)
    function dot_pullback(ΔΩ)
        dx = @thunk conj(ΔΩ) .* A.diag .* y  # A*y is this broadcast, can be fused
        dA = @thunk Diagonal(ΔΩ .* x .* conj(y))  # calculate N not N^2 elements
        dy = @thunk ΔΩ .* conj.(A.diag) .* x
        return (NO_FIELDS, dx, dA, dy)
    end
    dot_pullback(::Zero) = (NO_FIELDS, Zero(), Zero(), Zero())
    return z, dot_pullback
end

#####
##### `cross`
#####

function frule((_, Δa, Δb), ::typeof(cross), a::AbstractVector, b::AbstractVector)
    return cross(a, b), cross(Δa, b) .+ cross(a, Δb)
end

# TODO: support complex vectors
function rrule(::typeof(cross), a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    Ω = cross(a, b)
    function cross_pullback(ΔΩ)
        return (NO_FIELDS, @thunk(cross(b, ΔΩ)), @thunk(cross(ΔΩ, a)))
    end
    return Ω, cross_pullback
end


#####
##### `det`
#####

function frule((_, Δx), ::typeof(det), x::AbstractMatrix)
    Ω = det(x)
    # TODO Performance optimization: probably there is an efficent
    # way to compute this trace without during the full compution within
    return Ω, Ω * tr(x \ Δx)
end
frule((_, Δx), ::typeof(det), x::Number) = (det(x), Δx)

function rrule(::typeof(det), x::Union{Number, AbstractMatrix})
    Ω = det(x)
    function det_pullback(ΔΩ)
        ∂x = x isa Number ? ΔΩ : Ω * ΔΩ * inv(x)'
        return (NO_FIELDS, ∂x)
    end
    return Ω, det_pullback
end

#####
##### `logdet`
#####

function frule((_, Δx), ::typeof(logdet), x::Union{Number, StridedMatrix{<:Number}})
    Ω = logdet(x)
    return Ω, tr(x \ Δx)
end

function rrule(::typeof(logdet), x::Union{Number, StridedMatrix{<:Number}})
    Ω = logdet(x)
    function logdet_pullback(ΔΩ)
        ∂x = x isa Number ? ΔΩ / x' : ΔΩ * inv(x)'
        return (NO_FIELDS, ∂x)
    end
    return Ω, logdet_pullback
end

#####
##### `logabsdet`
#####

function frule((_, Δx), ::typeof(logabsdet), x::AbstractMatrix)
    Ω = logabsdet(x)
    (y, signy) = Ω
    b = tr(x \ Δx)
    ∂y = real(b)
    ∂signy = eltype(x) <: Real ? Zero() : im * imag(b) * signy
    ∂Ω = Composite{typeof(Ω)}(∂y, ∂signy)
    return Ω, ∂Ω
end

function rrule(::typeof(logabsdet), x::AbstractMatrix)
    Ω = logabsdet(x)
    function logabsdet_pullback(ΔΩ)
        (Δy, Δsigny) = ΔΩ
        (_, signy) = Ω
        f = signy' * Δsigny
        imagf = f - real(f)
        g = real(Δy) + imagf
        ∂x = g * inv(x)'
        return (NO_FIELDS, ∂x)
    end
    return Ω, logabsdet_pullback
end

#####
##### `trace`
#####

function frule((_, Δx), ::typeof(tr), x)
    return tr(x), tr(Δx)
end

function rrule(::typeof(tr), x)
    # This should really be a FillArray
    # see https://github.com/JuliaDiff/ChainRules.jl/issues/46
    function tr_pullback(ΔΩ)
        return (NO_FIELDS, Diagonal(fill(ΔΩ, size(x, 1))))
    end
    return tr(x), tr_pullback
end


#####
##### `pinv`
#####

@scalar_rule pinv(x) -(Ω ^ 2)

function frule(
    (_, Δx),
    ::typeof(pinv),
    x::AbstractVector{T},
    tol::Real = 0,
) where {T<:Union{Real,Complex}}
    y = pinv(x, tol)
    ∂y′ = sum(abs2, parent(y)) .* Δx .- 2real(y * Δx) .* parent(y)
    ∂y = y isa Transpose ? transpose(∂y′) : adjoint(∂y′)
    return y, ∂y
end

function frule(
    (_, Δx),
    ::typeof(pinv),
    x::LinearAlgebra.AdjOrTransAbsVec{T},
    tol::Real = 0,
) where {T<:Union{Real,Complex}}
    y = pinv(x, tol)
    ∂y = sum(abs2, y) .* vec(Δx') .- 2real(Δx * y) .* y
    return y, ∂y
end

# Formula for derivative adapted from Eq 4.12 of
# Golub, Gene H., and Victor Pereyra. "The Differentiation of Pseudo-Inverses and Nonlinear
# Least Squares Problems Whose Variables Separate."
# SIAM Journal on Numerical Analysis 10(2). (1973). 413-432. doi: 10.1137/0710036
function frule((_, ΔA), ::typeof(pinv), A::AbstractMatrix{T}; kwargs...) where {T}
    Y = pinv(A; kwargs...)
    m, n = size(A)
    # contract over the largest dimension
    if m ≤ n
        ∂Y = -Y * (ΔA * Y)
        ∂Y = add!!(∂Y, (ΔA' - Y * (A * ΔA')) * (Y' * Y))  # (I - Y A) ΔA' Y' Y
        ∂Y = add!!(∂Y, Y * (Y' * ΔA') * (I - A * Y))  # Y Y' ΔA' (I - A Y)
    else
        ∂Y = -(Y * ΔA) * Y
        ∂Y = add!!(∂Y, (I - Y * A) * (ΔA' * Y') * Y)  # (I - Y A) ΔA' Y' Y
        ∂Y = add!!(∂Y, (Y * Y') * (ΔA' - (ΔA' * A) * Y))  # Y Y' ΔA' (I - A Y)
    end
    return Y, ∂Y
end

function rrule(
    ::typeof(pinv),
    x::AbstractVector{T},
    tol::Real = 0,
) where {T<:Union{Real,Complex}}
    y = pinv(x, tol)
    function pinv_pullback(Δy)
        ∂x = sum(abs2, parent(y)) .* vec(Δy') .- 2real(y * Δy') .* parent(y)
        return (NO_FIELDS, ∂x, Zero())
    end
    return y, pinv_pullback
end

function rrule(
    ::typeof(pinv),
    x::LinearAlgebra.AdjOrTransAbsVec{T},
    tol::Real = 0,
) where {T<:Union{Real,Complex}}
    y = pinv(x, tol)
    function pinv_pullback(Δy)
        ∂x′ = sum(abs2, y) .* Δy .- 2real(y' * Δy) .* y
        ∂x = x isa Transpose ? transpose(conj(∂x′)) : adjoint(∂x′)
        return (NO_FIELDS, ∂x, Zero())
    end
    return y, pinv_pullback
end

function rrule(::typeof(pinv), A::AbstractMatrix{T}; kwargs...) where {T}
    Y = pinv(A; kwargs...)
    function pinv_pullback(ΔY)
        m, n = size(A)
        # contract over the largest dimension
        if m ≤ n
            ∂A = (Y' * -ΔY) * Y'
            ∂A = add!!(∂A, (Y' * Y) * (ΔY' - (ΔY' * Y) * A)) # Y' Y ΔY' (I - Y A)
            ∂A = add!!(∂A, (I - A * Y) * (ΔY' * Y) * Y') # (I - A Y) ΔY' Y Y'
        elseif m > n
            ∂A = Y' * (-ΔY * Y')
            ∂A = add!!(∂A, Y' * (Y * ΔY') * (I - Y * A)) # Y' Y ΔY' (I - Y A)
            ∂A = add!!(∂A, (ΔY' - A * (Y * ΔY')) * (Y * Y')) # (I - A Y) ΔY' Y Y'
        end
        return (NO_FIELDS, ∂A)
    end
    return Y, pinv_pullback
end

#####
##### `norm`
#####

function rrule(::typeof(norm), A::AbstractArray{<:Real}, p::Real=2)
    y = norm(A, p)
    function norm_pullback(ȳ)
        u = y^(1-p)
        ∂A = @thunk ȳ .* u .* abs.(A).^p ./ A
        ∂p = @thunk ȳ * (u * sum(a->abs(a)^p * log(abs(a)), A) - y * log(y)) / p
        (NO_FIELDS, ∂A, ∂p)
    end
    return y, norm_pullback
end

function rrule(::typeof(norm), x::Real, p::Real=2)
    function norm_pullback(ȳ)
        ∂x = @thunk ȳ * sign(x)
        ∂p = @thunk zero(x)  # TODO: should this be Zero()?
        (NO_FIELDS, ∂x, ∂p)
    end
    return norm(x, p), norm_pullback
end
