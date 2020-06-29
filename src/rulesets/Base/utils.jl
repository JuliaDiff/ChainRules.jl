# real(conj(x) * y) avoiding computing the imaginary part if possible
@inline _realconjtimes(x, y) = real(conj(x) * y)
@inline _realconjtimes(x::Complex, y::Complex) = real(x) * real(y) + imag(x) * imag(y)
@inline _realconjtimes(x::Real, y::Complex) = x * real(y)
@inline _realconjtimes(x::Complex, y::Real) = real(x) * y
@inline _realconjtimes(x::Real, y::Real) = x * y
