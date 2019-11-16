#####
##### `map`
#####

function rrule(::typeof(map), f, xs...)
    y = map(f, xs...)
    function map_pullback(ȳ)
        ntuple(length(xs)+2) do full_i
            full_i == 1 && return NO_FIELDS
            full_i == 2 && return DoesNotExist()
            i = full_i-2
            @thunk map(ȳ, xs...) do ȳi, xis...
                _, pullback = _checked_rrule(f, xis...)
                ∂xis = pullback(ȳi)
                extern(∂xis[i+1])  #+1 to skp ∂self
            end
        end
    end
    return y, map_pullback
end

#####
##### `mapreduce`, `mapfoldl`, `mapfoldr`
#####

for mf in (:mapreduce, :mapfoldl, :mapfoldr)
    sig = :(rrule(::typeof($mf), f, op, x::AbstractArray{<:Real}))
    call = :($mf(f, op, x))
    if mf === :mapreduce
        insert!(sig.args, 2, Expr(:parameters, Expr(:kw, :dims, :(:))))
        insert!(call.args, 2, Expr(:parameters, Expr(:kw, :dims, :dims)))
    end
    pullback_name = Symbol(mf, :_pullback)
    body = quote
        y = $call
        function $pullback_name(ȳ)
            ∂x = @thunk broadcast(x, ȳ) do xi, ȳi
                _, pullback_f = _checked_rrule(f, xi)
                _, ∂xi = pullback_f(ȳi)
                extern(∂xi)
            end
            (NO_FIELDS, DoesNotExist(), DoesNotExist(), ∂x)
        end
        return y, $pullback_name
    end
    eval(Expr(:function, sig, body))
end

#####
##### `sum`
#####

function frule(::typeof(sum), x)
    function sum_pushforward(_, ẋ)
        return sum(ẋ)
    end
    return sum(x), sum_pushforward
end

function rrule(::typeof(sum), x::AbstractArray{<:Real})
    function sum_pullback(ȳ)
        return (NO_FIELDS, @thunk(fill(ȳ, size(x))))
    end
    return sum(x), sum_pullback
end

function rrule(::typeof(sum), f, x::AbstractArray{<:Real}; dims=:)
    y, mr_pullback = rrule(mapreduce, f, Base.add_sum, x; dims=dims)
    function sum_pullback(ȳ)
        return NO_FIELDS, DoesNotExist(), last(mr_pullback(ȳ))
    end
    return y, sum_pullback
end

function rrule(::typeof(sum), x::AbstractArray{<:Real}; dims=:)
    y,  inner_pullback = rrule(sum, identity, x; dims=dims)
    function sum_pullback(ȳ)
        return NO_FIELDS, last(inner_pullback(ȳ))
    end
    return y, sum_pullback
end

function rrule(::typeof(sum), ::typeof(abs2), x::AbstractArray{<:Real}; dims=:)
    y = sum(abs2, x; dims=dims)
    function sum_abs2_pullback(ȳ)
        return (NO_FIELDS, DoesNotExist(), @thunk(2ȳ .* x))
    end
    return y, sum_abs2_pullback
end
