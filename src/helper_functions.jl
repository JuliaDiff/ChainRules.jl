# Special purpose updating for operations which can be done in-place. This function is
# just internal and free-form; it is not a method of `accumulate!` directly as it does
# not adhere to the expected method signature form, i.e. `accumulate!(value, rule, args)`.
# Instead it's `_update!(old, new, extrastuff...)` and is not specific to any particular
# rule.

_update!(x, y) = x + y
_update!(x::Array{T,N}, y::AbstractArray{T,N}) where {T,N} = x .+= y

_update!(x, ::Zero) = x
_update!(::Zero, y) = y
_update!(::Zero, ::Zero) = Zero()

function _update!(x::NamedTuple{Ns}, y::NamedTuple{Ns}) where Ns
    return NamedTuple{Ns}(map(p->_update!(getproperty(x, p), getproperty(y, p)), Ns))
end

function _update!(x::NamedTuple, y, p::Symbol)
    new = NamedTuple{(p,)}((_update!(getproperty(x, p), y),))
    return merge(x, new)
end

function _update!(x::NamedTuple{Ns}, y::NamedTuple{Ns}, p::Symbol) where Ns
    return _update!(x, getproperty(y, p), p)
end


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
