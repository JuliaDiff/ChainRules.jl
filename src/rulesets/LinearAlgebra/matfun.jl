# matrix functions of dense matrices

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

