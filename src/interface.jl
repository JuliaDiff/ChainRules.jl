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
##### `Seed`
#####

struct Seed{V}
    value::V
    store_into::Bool
    increment_adjoint::Bool
    materialize::Bool
end

function Seed(value;
              store_into::Bool = false,
              increment_adjoint::Bool = true,
              materialize::Bool = true)
    return Seed(value, store_into, increment_adjoint, materialize)
end

materialize_via_seed(::Nothing, partial) = materialize(partial)

function materialize_via_seed(seed::Seed, partial)
    if seed.materialize
        if seed.store_into
            return materialize!(seed.value, partial)
        else
            return materialize(partial)
        end
    else
        return partial
    end
end

#####
##### `fchain`
#####

function fchain(args...)
    seed, partial = _fchain(nothing, args...)
    return materialize_via_seed(seed, partial)
end

@inline _fchain(seed, ẋ, ∂::Thunk) = (seed, broadcasted(*, ẋ, ∂()))
@inline _fchain(seed, ẋ::Nothing, ∂::Thunk) = (seed, false)
@inline _fchain(seed::Nothing, ẋ::Seed, ∂::Thunk) = (ẋ, broadcasted(*, ẋ.value, ∂()))

@inline function _fchain(seed::Seed, ẋ::Seed, ∂::Thunk)
    error("`fchain` does not support multiple simultaneous `Seed` arguments")
end

@inline function _fchain(seed, ẋ, ∂::Thunk, args...)
    seed, partial = _fchain(seed, ẋ, ∂)
    return broadcasted(+, partial, _fchain(seed, args...))
end

#####
##### `rchain`
#####

@inline rchain(x̄::Nothing, ∂::Thunk) = false

rchain(x̄::Seed, ∂::Thunk) = _rchain(x̄, x̄.value, ∂())
rchain(x̄, ∂::Thunk) = _rchain(nothing, x̄, ∂())

function _rchain(seed, x̄, partial)
    if !isa(seed, Seed) || seed.increment_adjoint
        partial = broadcasted(+, x̄, partial)
    end
    return materialize_via_seed(seed, partial)
end
