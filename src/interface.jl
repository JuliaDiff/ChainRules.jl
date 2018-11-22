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
    use_as_storage::Bool
    dont_materialize::Bool
    has_single_dependent::Bool
end

function Seed(value;
              use_as_storage::Bool = false,
              dont_materialize::Bool = false,
              has_single_dependent::Bool = false)
    return Seed(value, use_as_storage, dont_materialize, has_single_dependent)
end

materialize_via_seed(::Nothing, partial) = materialize(partial)

function materialize_via_seed(seed::Seed, partial)
    if seed.dont_materialize
        return partial
    else
        if seed.use_as_result_storage
            return materialize!(seed.value, partial)
        else
            return materialize(partial)
        end
    end
end

#####
##### `fchain`
#####

function fchain(args...)
    seed, partial = _fchain(nothing, args...)
    return materialize_via_seed(seed, partial)
end

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
    if isa(seed, Seed) && seed.has_single_dependent
        return materialize_via_seed(seed, partial)
    end
    return materialize_via_seed(seed, broadcasted(+, x̄, partial))
end
