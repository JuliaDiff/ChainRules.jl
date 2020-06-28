# real(x * conj(y)) avoiding computing the imaginary part
_realconjtimes(x, y) = real(x) * real(y) + imag(x) * imag(y)
_realconjtimes(x::Real, y) = x * real(y)
_realconjtimes(x, y::Real) = real(x) * y
_realconjtimes(x::Real, y::Real) = x * y
