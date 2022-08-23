using JLArrays  # provides a fake GPU array
JLArrays.allowscalar(false)

using Adapt
jl32(xs) = adapt(JLArray{Float32}, xs)  # this works much like `CUDA.cu`
# This defines the behaviour of `adapt(JLArray{Float32}, xs)` on arrays.
# This is piracy, but only while running these tests... could avoid by defining a struct.
Adapt.adapt_storage(::Type{<:JLArray{Float32}}, xs::AbstractArray{<:Complex}) = convert(JLArray{ComplexF32}, xs)
Adapt.adapt_storage(::Type{<:JLArray{Float32}}, x::Float64) = Float32(x)
Adapt.adapt_storage(::Type{<:JLArray{Float32}}, x::ComplexF64) = ComplexF32(x)
Adapt.adapt(T::Type{<:JLArray{Float32}}, x::AbstractThunk) = adapt(T, unthunk(x))

f32(xs) = adapt(Array{Float32}, xs)
# This maps things back to the CPU. Or, applied to CPU arrays, changes to Float32 to match GPU calculation.
Adapt.adapt_storage(::Type{<:Array{Float32}}, xs::AbstractArray{<:Complex}) = convert(Array{ComplexF32}, xs)
Adapt.adapt_storage(::Type{<:Array{Float32}}, x::Float64) = Float32(x)
Adapt.adapt_storage(::Type{<:Array{Float32}}, x::ComplexF64) = ComplexF32(x)
Adapt.adapt(T::Type{<:Array{Float32}}, x::AbstractThunk) = adapt(T, unthunk(x))
    

"""
    @gpu test_rrule(sum, rand(3))
    @gpu test_frule((0, rand(3)), sum, rand(3))

After running the test shown, this converts all arrays to `GPUArray`s,
and checks that the rule accepts this, and produces the same result.
Uses the mock GPU array from JLArrays.jl, even if you have a real GPU available.
Does not understand all features of `test_rrule`, in particular `⊢`.

  @gpu rrule(sum, rand(3))
  @gpu frule((0, rand(3)), sum, rand(3))
  
Used directly on a rule, this compares a CPU to a GPU run, without
checking correctness. Both runs will convert numbers to `Float32` first.
"""
macro gpu(ex)
    _gpu_macro(ex, false, __source__)
end
macro gpu_broken(ex)
    _gpu_macro(ex, true, __source__)
end 
function _gpu_macro(ex, broken::Bool, __source__)
    Meta.isexpr(ex, :call) && ex.args[1] in (:test_rrule, :test_frule, :rrule, :frule) ||
      error("@gpu doesn't understand this input")
    ex2 = if Meta.isexpr(ex.args[2], :parameters)
        :($_gpu_test($(ex.args[2]), $(ex.args[1]), $(ex.args[3:end]...)))
    else
        :($_gpu_test($(ex.args[1]), $(ex.args[2:end]...)))
    end
    return if broken
        Expr(:block, ex, Expr(:macrocall, Symbol("@test_broken"), __source__, ex2)) |> esc
    else
        Expr(:block, ex, Expr(:macrocall, Symbol("@test"), __source__, ex2)) |> esc
        # NB this @test should report a line number in your tests, not here.
    end
end

function _gpu_test(::typeof(test_rrule), xs...; fkwargs = (;), kw...)
    _gpu_test(rrule, xs...; fkwargs...)
end

function _gpu_test(::typeof(test_frule), xs...; fkwargs = (;), kw...)
    _gpu_test(frule, xs...; fkwargs...)
end

function _gpu_test(::typeof(rrule), xs...; kw...)
    y, bk = rrule(f32(xs)...; kw...)

    eltype(y) in (Float64, ComplexF64) && return false  # test for accidental Float64 promotion

    y1 = one.(y)  # crude way to get input sensitivity
    dxs = unthunk.(bk(y1))

    gpu_y, gpu_bk = rrule(jl32(xs)...; kw...)
    gpu_dxs = adapt(Array, unthunk.(gpu_bk(one.(gpu_y))))

    agree = map(gpu_dxs, dxs) do a, b
        a isa AbstractZero && return b isa AbstractZero
        isapprox(a, b)  # compare on CPU to avoid error from e.g. `isapprox(jl(x), jl(permutedims(x))')`
    end
    return all(agree)
    # NB this does not contain @test, it just returns true/false. Then it can be used with `@test_broken` without going mad.
end

function _gpu_test(::typeof(frule), xdots, f::Function, xs...; kw...)
    y = frule(f32(xdots), f, f32(xs)...; kw...)
    @test eltype(y[2]) ∉ (Float64, ComplexF64)
    gpu_y = frule(jl32(xdots), f, jl32(xs)...; kw...)
    ChainRulesTestUtils.test_approx(gpu_y, jl32(y))
    true
end
function _gpu_test(::typeof(frule), f::Function, xs...; kw...)
    xdots = (NoTangent(), xs...)  # a pretty crude way to generate xdots for you
    _gpu_test(frule, xdots, f, xs...; kw...)
end
function _gpu_test(::typeof(frule), f::Function, g::Function, xs...; kw...)  # solves an ambiguity
    xdots = (NoTangent(), NoTangent(), xs...)
    _gpu_test(frule, xdots, f, g, xs...; kw...)
end

const CFG = ChainRulesTestUtils.TestConfig()

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

"A version of `*` with only an `frule` defined"
fstar(A, B) = A * B
ChainRulesCore.frule((_, ΔA, ΔB), ::typeof(fstar), A, B) = A * B, muladd(ΔA, B, A * ΔB)

"A version of `log` with only an `frule` defined"
flog(x::Number) = log(x)
ChainRulesCore.frule((_, Δx), ::typeof(flog), x::Number) = log(x), inv(x) * Δx

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
    
    @testset "fstar, flog" begin
        test_frule(fstar, 1.2, 3.4 + 5im)
        test_frule(flog, 6.7)
        test_frule(flog, 8.9 + im)    
    end

    @testset "ambiguities" begin
        @test [] == filter(Test.detect_ambiguities(ChainRules, ChainRulesCore)) do t
            (t[1].name in [:rrule, Symbol("rrule##kw")])
        end
    end
end
