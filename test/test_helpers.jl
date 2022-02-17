"""
    Multiplier(x)

Stores a fixed `x` and multiplies by it, especially for testing
the gradient of higher order functions with respect to `x`.
```
julia> map(Multiplier(pi), [1 10 100 1000])
1×4 Matrix{Float64}:
 3.14159  31.4159  314.159  3141.59

julia> map(Multiplier(2), [1 10 100], [-1 3 -7])  # two arguments
1×3 Matrix{Int64}:
 -2  60  -1400

julia> map(Multiplier([1 2; 3 4]), ([5, 6], [7, 8]))  # x isa Matrix
([17, 39], [23, 53])
```
"""
struct Multiplier{T}
    x::T
end
(m::Multiplier)(y) = m.x * y
(m::Multiplier)(y, z) = (m.x * y) * z

function ChainRulesCore.rrule(m::Multiplier, y)
    Multiplier_pullback(dΩ) = (Tangent{typeof(m)}(; x = dΩ * y'), m.x' * dΩ)
    return m(y), Multiplier_pullback
end
function ChainRulesCore.rrule(m::Multiplier, y, z)
    Multiplier_pullback_3(dΩ) = (
        Tangent{typeof(m)}(; x = dΩ * (y * z)'),
        m.x' * dΩ * z', 
        (m.x * y)' * dΩ,
    )
    return m(y, z), Multiplier_pullback_3
end

"""
    Counter()

Multiplies its input by number that increments on each call,
for testing execution order. Has a gradient `rrule` which
similarly increases by 10 each call, this is *not* the true gradient!
```
julia> map(Counter(), [1 1 1 10 10 10])
1×6 Matrix{Int64}:
 1  2  3  40  50  60
```
"""
mutable struct Counter
    n::Int
    Counter(n=0) = new(n)
end
Base.:(==)(a::Counter, b::Counter) = (a.n == b.n)
(c::Counter)(x) = x * (c.n += 1) 
(c::Counter)(x, y) = (x + y) * (c.n += 1)

function ChainRulesCore.rrule(c::Counter, x)
    # False gradient, again just to record sequence of calls:
    Counter_back(dΩ) = (NoTangent(), dΩ * (c.n += 10))
    return c(x), Counter_back
end
function ChainRulesCore.rrule(c::Counter, x, y)
    Counter_back_2(dΩ) = (NoTangent(), dΩ * (c.n += 10), dΩ * (c.n += 10))
    return c(x, y), Counter_back_2
end

# NoRules - has no rules defined
struct NoRules; end

"A function that outputs a vector from a scalar for testing"
make_two_vec(x) = [x, x]
function ChainRulesCore.rrule(::typeof(make_two_vec), x)
    make_two_vec_pullback(ȳ) = (NoTangent(), sum(ȳ))
    return make_two_vec(x), make_two_vec_pullback
end

# Minimal quaternion implementation for testing rules that accept numbers that don't commute
# under multiplication
# adapted from Base Julia
# https://github.com/JuliaLang/julia/blob/bb5b98e72a151c41471d8cc14cacb495d647fb7f/test/testhelpers/Quaternions.jl

export Quaternion

struct Quaternion{T<:Real} <: Number
    s::T
    v1::T
    v2::T
    v3::T
end
const QuaternionF64 = Quaternion{Float64}
Quaternion(s::Real, v1::Real, v2::Real, v3::Real) = Quaternion(promote(s, v1, v2, v3)...)
Quaternion{T}(s::Real) where {T} = Quaternion(T(s), zero(T), zero(T), zero(T))
Base.convert(::Type{Quaternion{T}}, s::Real) where {T <: Real} =
    Quaternion{T}(convert(T, s), zero(T), zero(T), zero(T))
Base.promote_rule(::Type{Quaternion{S}}, ::Type{T}) where {S<:Real,T<:Real} = Quaternion{Base.promote_type(S,T)}
Base.abs2(q::Quaternion) = q.s*q.s + q.v1*q.v1 + q.v2*q.v2 + q.v3*q.v3
Base.float(z::Quaternion{T}) where T = Quaternion(float(z.s), float(z.v1), float(z.v2), float(z.v3))
Base.abs(q::Quaternion) = sqrt(abs2(q))
Base.real(::Type{Quaternion{T}}) where {T} = T
Base.real(q::Quaternion) = q.s
Base.conj(q::Quaternion) = Quaternion(q.s, -q.v1, -q.v2, -q.v3)
Base.isfinite(q::Quaternion) = isfinite(q.s) & isfinite(q.v1) & isfinite(q.v2) & isfinite(q.v3)
Base.zero(::Type{Quaternion{T}}) where T = Quaternion{T}(zero(T), zero(T), zero(T), zero(T))
Base.:(+)(ql::Quaternion, qr::Quaternion) =
 Quaternion(ql.s + qr.s, ql.v1 + qr.v1, ql.v2 + qr.v2, ql.v3 + qr.v3)
Base.:(-)(ql::Quaternion, qr::Quaternion) =
 Quaternion(ql.s - qr.s, ql.v1 - qr.v1, ql.v2 - qr.v2, ql.v3 - qr.v3)
Base.:(-)(q::Quaternion) = Quaternion(-q.s, -q.v1, -q.v2, -q.v3)
Base.:(*)(q::Quaternion, w::Quaternion) = Quaternion(q.s*w.s - q.v1*w.v1 - q.v2*w.v2 - q.v3*w.v3,
                                            q.s*w.v1 + q.v1*w.s + q.v2*w.v3 - q.v3*w.v2,
                                            q.s*w.v2 - q.v1*w.v3 + q.v2*w.s + q.v3*w.v1,
                                            q.s*w.v3 + q.v1*w.v2 - q.v2*w.v1 + q.v3*w.s)
Base.:(*)(q::Quaternion, r::Real) = Quaternion(q.s*r, q.v1*r, q.v2*r, q.v3*r)
Base.:(*)(q::Quaternion, b::Bool) = b * q # remove method ambiguity
Base.:(/)(q::Quaternion, w::Quaternion) = q * conj(w) * (1.0 / abs2(w))
Base.:(\)(q::Quaternion, w::Quaternion) = conj(q) * w * (1.0 / abs2(q))
function Base.rand(rng::AbstractRNG, ::Random.SamplerType{Quaternion{T}}) where {T<:Real}
    return Quaternion{T}(rand(rng, T), rand(rng, T), rand(rng, T), rand(rng, T))
end
function Base.randn(rng::AbstractRNG, ::Type{Quaternion{T}}) where {T<:AbstractFloat}
    return Quaternion{T}(
        randn(rng, T) / 2,
        randn(rng, T) / 2,
        randn(rng, T) / 2,
        randn(rng, T) / 2,
    )
end
(project::ProjectTo{<:Real})(dx::Quaternion) = project(real(dx))
function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, q::Quaternion{Float64})
    return Quaternion(rand(rng, -9:0.1:9), rand(rng, -9:0.1:9), rand(rng, -9:0.1:9), rand(rng, -9:0.1:9))
end

@testset "test_helpers.jl" begin

    @testset "Multiplier" begin
        # One argument
        test_rrule(Multiplier(4.0), 3.0)
        test_rrule(Multiplier(5.0 + 6im), 7.0 + 8im)
        test_rrule(Multiplier(rand(2,3)), rand(3,4))

        # Two arguments
        test_rrule(Multiplier(1.2), 3.4, 5.6)
        test_rrule(Multiplier(1.0 + 2im), 3.0 + 4im, 5.0 - 6im)
        test_rrule(Multiplier(rand(2,3)), rand(3,4), rand(4,5))
    end

    @testset "Counter" begin
        c = Counter()
        @test map(c, ones(5)) == 1:5
        @test c == Counter(5)
        y, back = rrule(c, 666)
        @test back(1) == (NoTangent(), 16)
        @test back(10) == (NoTangent(), 260)
        y2, back2 = rrule(c, 777, 888)
        @test back2(1) == (NoTangent(), 37, 47)
    end

    @testset "make_two_vec" begin
        test_rrule(make_two_vec, 1.5)
    end

end