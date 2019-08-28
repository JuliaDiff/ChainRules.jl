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
    return y, (NO_FIELDS_RULE, DNERule(), ∂xs...)
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
        return y, (NO_FIELDS_RULE, DNERule(), DNERule(), ∂x)
    end
    eval(Expr(:function, sig, body))
end

#####
##### `sum`
#####

frule(::typeof(sum), x) = (sum(x), (ZERO_RULE, Rule(sum)))

rrule(::typeof(sum), x) = (sum(x), (NO_FIELDS_RULE, Rule(cast)))

function rrule(::typeof(sum), f, x::AbstractArray{<:Real}; dims=:)
    y, (_, _, ∂x) = rrule(mapreduce, f, Base.add_sum, x; dims=dims)
    return y, (NO_FIELDS_RULE, DNERule(), ∂x)
end

function rrule(::typeof(sum), x::AbstractArray{<:Real}; dims=:)
    y, (no_fields, _, ∂x) = rrule(sum, identity, x; dims=dims)
    @assert(no_fields === NO_FIELDS_RULE)
    return y, (NO_FIELDS_RULE, ∂x)
end

function rrule(::typeof(sum), ::typeof(abs2), x::AbstractArray{<:Real}; dims=:)
    y = sum(abs2, x; dims=dims)
    ∂x = Rule(ȳ -> 2ȳ .* x)
    return y, (NO_FIELDS_RULE, DNERule(), ∂x)
end
