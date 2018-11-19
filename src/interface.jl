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
    casted = should_increment(x̄) ? broadcasted(+, x̄_value, thunk) : thunk
    if should_materialize_into(x̄)
        return materialize!(x̄_value, casted)
    else
        return materialize(casted)
    end
end

adjoint_value(x̄) = x̄

should_increment(::Any) = true

should_materialize_into(::Any) = false

#####
##### miscellaneous utilities
#####

# TODO: More defaults, obviously!

markup(::Any) = Ignore()
markup(::Real) = RealScalar()
markup(::Complex) = ComplexScalar()
markup(x::Tuple{Vararg{<:Real}}) = RealTensor()
markup(x::Tuple{Vararg{<:Complex}}) = ComplexTensor()
markup(x::AbstractArray{<:Real}) = RealTensor()
markup(x::AbstractArray{<:Complex}) = ComplexTensor()
markup(x::AbstractArray) = error("Cannot infer domain of array from eltype", x)

struct CPU end

device(::AbstractArray) = CPU()

ismutable(::Array) = true

ismutable(::AbstractArray) = false

should_materialize_into(::Array) = true
