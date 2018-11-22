#####
##### `Thunk`
#####

macro thunk(body)
    return :(Thunk(() -> $(esc(body))))
end

struct Thunk{F}
    f::F
end

@inline (thunk::Thunk{F})() where {F} = (thunk.f)()

#####
##### `fchain`
#####

fchain(args...) = materialize(_fchain(args...))

@inline _fchain(ẋ::Nothing, ∂::Thunk) = false
@inline _fchain(ẋ, ∂::Thunk) = broadcasted(*, ẋ, ∂())
_fchain(ẋ, ∂::Thunk, args...) = broadcasted(+, _fchain(ẋ, ∂), _fchain(args...))

#####
##### `rchain!`
#####

@inline rchain!(x̄::Nothing, ∂::Thunk) = false

@inline function rchain!(x̄, ∂::Thunk)
    thunk = ∂()
    x̄_value = adjoint_value(x̄)
    casted = should_increment_adjoint(x̄) ? broadcasted(+, x̄_value, thunk) : thunk
    if should_materialize_adjoint_in_place(x̄)
        return materialize!(x̄_value, casted)
    else
        return materialize(casted)
    end
end

adjoint_value(x̄) = x̄

should_increment_adjoint(::Any) = true

should_materialize_adjoint_in_place(::Any) = false
should_materialize_adjoint_in_place(::Array) = true
