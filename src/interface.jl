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
##### `forward_chain`
#####

forward_chain(args...) = materialize(_forward_chain(args...))

@inline _forward_chain(ẋ::Nothing, ∂::Thunk) = false
@inline _forward_chain(ẋ, ∂::Thunk) = broadcasted(*, ẋ, ∂())
_forward_chain(ẋ, ∂::Thunk, args...) = broadcasted(+, _forward_chain(ẋ, ∂), _forward_chain(args...))

#####
##### `reverse_chain!`
#####

@inline reverse_chain!(x̄::Nothing, ∂::Thunk) = false

@inline function reverse_chain!(x̄, ∂::Thunk)
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
