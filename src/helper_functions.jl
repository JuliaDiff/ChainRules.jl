# Internal helpers for defining the `add!` field of an `InplaceableThunk`

_update!(x, y) = x + y
_update!(x::Array{T,N}, y::AbstractArray{T,N}) where {T,N} = x .+= y

_update!(x, ::Zero) = x
_update!(::Zero, y) = y
_update!(::Zero, ::Zero) = Zero()


function _update!(x::NamedTuple, y, p::Symbol)
    y = extern(y)
    yp = getproperty(y, p)
    xp = getproperty(x, p)
    new_xp = _update!(xp, yp)
    new = NamedTuple{(p,)}((new_xp,))
    return merge(x, new)
end

#==
function _update!(x::NamedTuple{Ns}, y::NamedTuple{Ns}) where Ns
    return NamedTuple{Ns}(map(p->_update!(getproperty(x, p), getproperty(y, p)), Ns))
end


function _update!(x::NamedTuple{Ns}, y::NamedTuple{Ns}, p::Symbol) where Ns
    return _update!(x, getproperty(y, p), p)
end
==#

"""
    _checked_rrule

like `rrule` but throws an error if the `rrule` is not defined.
Rather than returning `nothing`
"""
function _checked_rrule(f, args...; kwargs...)
    r = rrule(f, args...; kwargs...)
    r isa Nothing && _throw_checked_rrule_error(f, args...; kwargs...)
    return r
end


@noinline function _throw_checked_rrule_error(f, args...; kwargs...)
    io = IOBuffer()
    print(io, "can't differentiate `", f, '(')
    join(io, map(arg->string("::", typeof(arg)), args), ", ")
    if !isempty(kwargs)
        print(io, ";")
        join(io, map(((k, v),)->string(k, "=", v), kwargs), ", ")
    end
    print(io, ")`; no matching `rrule` is defined")
    throw(ArgumentError(String(take!(io))))
end
