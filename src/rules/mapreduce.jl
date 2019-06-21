#####
##### `map`
#####

function rrule(::typeof(map), f, xs...)
    y = map(f, xs...)
    ∂xs = ntuple(length(xs)) do i
        Rule() do ȳ
            map(ȳ, xs...) do ȳi, xis...
                _, ∂xis = _checked_rrule(f, xis...)
                extern(∂xis[i](ȳi))
            end
        end
    end
    return y, (DNERule(), ∂xs...)
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
    body = quote
        y = $call
        ∂x = Rule() do ȳ
            broadcast(x, ȳ) do xi, ȳi
                _, ∂xi = _checked_rrule(f, xi)
                extern(∂xi(ȳi))
            end
        end
        return y, (DNERule(), DNERule(), ∂x)
    end
    eval(Expr(:function, sig, body))
end

#####
##### `sum`
#####

frule(::typeof(sum), x) = (sum(x), Rule(sum))

rrule(::typeof(sum), x) = (sum(x), Rule(cast))

function rrule(::typeof(sum), f, x::AbstractArray{<:Real}; dims=:)
    y, (_, _, ∂x) = rrule(mapreduce, f, Base.add_sum, x; dims=dims)
    return y, (DNERule(), ∂x)
end

function rrule(::typeof(sum), x::AbstractArray{<:Real}; dims=:)
    y, (_, ∂x) = rrule(sum, identity, x; dims=dims)
    return y, ∂x
end

function rrule(::typeof(sum), ::typeof(abs2), x::AbstractArray{<:Real}; dims=:)
    y = sum(abs2, x; dims=dims)
    ∂x = Rule(ȳ -> 2ȳ .* x)
    return y, (DNERule(), ∂x)
end

#####
##### `mean`
#####

_denom(x, dims::Colon) = length(x)
_denom(x, dims::Integer) = size(x, dims)
_denom(x, dims) = mapreduce(i->size(x, i), Base.mul_prod, unique(dims), init=1)

# TODO: We have `mean(f, x; dims)` as of 1.3.0-DEV.36

function rrule(::typeof(mean), x::AbstractArray{<:Real}; dims=:)
    _, dx = rrule(sum, x; dims=dims)
    n = _denom(x, dims)
    return mean(x; dims=dims), Rule(ȳ -> dx(ȳ) / n)
end

function rrule(::typeof(mean), f, x::AbstractArray{<:Real})
    _, (_, dx) = rrule(sum, f, x)
    n = _denom(x, :)
    return mean(f, x), (DNERule(), Rule(ȳ -> dx(ȳ) / n))
end
