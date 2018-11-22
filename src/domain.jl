#####
##### `AbstractDomain`
#####

abstract type AbstractDomain end

struct RealDomain <: AbstractDomain end

struct ComplexDomain <: AbstractDomain end

struct IgnoreDomain <: AbstractDomain end

#####
##### `DomainSignature`
#####

struct DomainSignature{I <: Tuple{Vararg{AbstractDomain}},
                       O <: Tuple{Vararg{AbstractDomain}}}
    input::I
    output::O
end

DomainSignature(input, output) = DomainSignature(_domain.(tuplify(input)),
                                                 _domain.(tuplify(output)))

tuplify(x) = tuple(x)
tuplify(x::Tuple) = x

_domain(x::AbstractDomain) = x
_domain(x) = domain(x)

# TODO: overloads for more types (e.g. StaticArrays)
domain(::Any) = IgnoreDomain()
domain(::Real) = RealDomain()
domain(::Complex) = ComplexDomain()
domain(x::AbstractArray{<:Real}) = RealDomain()
domain(x::AbstractArray{<:Complex}) = ComplexDomain()
domain(x::AbstractArray) = error("Cannot infer domain of this array from its eltype: ", x)

# TODO: Should `DomainSignature` be changed to support destructured tuple
# elements, e.g. `@domain({R×(R×C×(R...))×(R...) → R})`?
domain(x::Tuple{Vararg{<:Real}}) = RealDomain()
domain(x::Tuple{Vararg{<:Complex}}) = ComplexDomain()

#####
##### `@domain`
#####

macro domain(expr)
    domain_from_signature(expr)
end

const MALFORMED_SIG_ERROR_MESSAGE = "Malformed expression given to `@domain`; see `@domain` docstring for proper format."

const CARTESIAN_PRODUCT_SYMBOL = :×

function domain_from_signature(expr)
    is_type_requested = Meta.isexpr(expr, :braces)
    if is_type_requested
        expr = first(expr.args)
    end
    @assert(Meta.isexpr(expr, :call) && expr.args[1] === :→ && length(expr.args) === 3, MALFORMED_SIG_ERROR_MESSAGE)
    parse_function = x -> parse_into_domain_type(x, is_type_requested)
    inputs = map(parse_function, split_infix_args(expr.args[2], CARTESIAN_PRODUCT_SYMBOL))
    outputs = map(parse_function, split_infix_args(expr.args[3], CARTESIAN_PRODUCT_SYMBOL))
    if is_type_requested
        return :(DomainSignature{<:Tuple{$(inputs...)}, <:Tuple{$(outputs...)}})
    else
        return :(DomainSignature(($(inputs...),), ($(outputs...),)))
    end
end

split_infix_args(invocation::Symbol, ::Symbol) = (invocation,)

function split_infix_args(invocation::Expr, op::Symbol)
    if Meta.isexpr(invocation, :call) && invocation.args[1] === op
        return (split_infix_args(invocation.args[2], op)..., invocation.args[3])
    end
    return (invocation,)
end

function parse_into_domain_type(x, is_type_requested::Bool)
    if x === :R
        result = :(RealDomain)
    elseif x === :C
        result = :(ComplexDomain)
    elseif x === :_
        result = :(IgnoreDomain)
    elseif Meta.isexpr(x, :(...))
        if is_type_requested
            result = :(Vararg{$(parse_into_domain_type(first(x.args), is_type_requested))})
        else
            error("`@domain` only supports varargs when a type is requested; ",
                  "try `@domain({...})` instead of `@domain(...)`.")
        end
    else
        error("Encountered unparseable domain signature element `", x, "`. ",
              MALFORMED_SIG_ERROR_MESSAGE)
    end
    return is_type_requested ? result : Expr(:call, result)
end
