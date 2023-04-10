using Base.Broadcast: broadcasted

if VERSION < v"1.7"
    Base.ndims(::Type{<:AbstractArray{<:Any,N}}) where {N} = N
end
BS0 = Broadcast.BroadcastStyle(Float64)
BS1 = Broadcast.BroadcastStyle(Vector)  # without ndims method, error on 1.6
BS2 = Broadcast.BroadcastStyle(Matrix)

BT1 = Broadcast.BroadcastStyle(Tuple)

@testset "Broadcasting" begin
    @testset "split 1: trivial path" begin
        # test_rrule(copy∘broadcasted, >, rand(3), rand(3))  # MethodError: no method matching eps(::UInt64) inside FiniteDifferences
        y1, bk1 = rrule(CFG, copy∘broadcasted, BS1, >, rand(3), rand(3))
        @test y1 isa AbstractArray{Bool}
        @test all(d -> d isa AbstractZero, bk1(99))

        y2, bk2 = rrule(CFG, copy∘broadcasted, BT1, isinteger, Tuple(rand(3)))
        @test y2 isa Tuple{Bool,Bool,Bool}
        @test all(d -> d isa AbstractZero, bk2(99))
    end

    @testset "split 2: derivatives" begin
        test_rrule(copy∘broadcasted, BS1, log, rand(3) .+ 1)
        # `check_inferred` doesn't accept the `Union` returned from ProjectTo as of
        # ChainRuleCore 1.15.4 https://github.com/JuliaDiff/ChainRulesCore.jl/issues/586
        test_rrule(copy∘broadcasted, BT1, log, Tuple(rand(3) .+ 1); check_inferred=false)

        # Two args uses StructArrays
        test_rrule(copy∘broadcasted, BS1, atan, rand(3), rand(3))
        test_rrule(copy∘broadcasted, BS2, atan, rand(3), rand(4)')
        test_rrule(copy∘broadcasted, BS1, atan, rand(3), rand())
        test_rrule(copy∘broadcasted, BT1, atan, rand(3), Tuple(rand(1)))
        test_rrule(copy∘broadcasted, BT1, atan, Tuple(rand(3)), Tuple(rand(3)), check_inferred = VERSION > v"1.7")

        # test_rrule(copy∘broadcasted, *, BS1, rand(3), Ref(rand()))  # don't know what I was testing
    end

    @testset "split 3: forwards" begin
        # In test_helpers.jl, `flog` and `fstar` have only `frule`s defined, nothing else.
        test_rrule(copy∘broadcasted, BS1, flog, rand(3))
        @test_skip test_rrule(copy∘broadcasted, BS1, flog, rand(3) .+ im)  # not OK, assumed analyticity, fixed in PR710
        # Also, `sin∘cos` may use this path as CFG uses frule_via_ad
        # TODO use different CFGs, https://github.com/JuliaDiff/ChainRulesTestUtils.jl/issues/255
    end

    @testset "split 4: generic" begin
        test_rrule(copy∘broadcasted, BS1, sin∘cos, rand(3), check_inferred=false)
        test_rrule(copy∘broadcasted, BS2, sin∘atan, rand(3), rand(3)', check_inferred=false)
        test_rrule(copy∘broadcasted, BS1, sin∘atan, rand(), rand(3), check_inferred=false)
        test_rrule(copy∘broadcasted, BS1, ^, rand(3), 3.0, check_inferred=false)  # NoTangent vs. Union{NoTangent, ZeroTangent}
        # Many have quite small inference failures, like:
        # return type Tuple{NoTangent, NoTangent, Vector{Float64}, Float64} does not match inferred
        #  return type Tuple{NoTangent, Union{NoTangent, ZeroTangent}, Vector{Float64}, Float64}

        # From test_helpers.jl
        test_rrule(copy∘broadcasted, BS1, Multiplier(rand()), rand(3), check_inferred=false)
        test_rrule(copy∘broadcasted, BS2, Multiplier(rand()), rand(3), rand(4)', check_inferred=false)  # Union{ZeroTangent, Tangent{Multiplier{...
        @test_skip test_rrule(copy∘broadcasted, BS1, Multiplier(rand()), rand(3), 5.0im, check_inferred=false)  # ProjectTo(f) fails to remove the imaginary part of Multiplier's gradient
        test_rrule(copy∘broadcasted, BS1, make_two_vec, rand(3), check_inferred=false)

        # Non-diff components -- note that with BroadcastStyle, Ref is from e.g. Broadcast.broadcastable(nothing)
        test_rrule(copy∘broadcasted, BS2, first∘tuple, rand(3), Ref(:sym), rand(4)', check_inferred=false)
        test_rrule(copy∘broadcasted, BS2, last∘tuple, rand(3), Ref(nothing), rand(4)', check_inferred=false)
        test_rrule(copy∘broadcasted, BS1, |>, rand(3), Ref(sin), check_inferred=false)
        _call(f, x...) = f(x...)
        test_rrule(copy∘broadcasted, BS2, _call, Ref(atan), rand(3), rand(4)', check_inferred=false)

        test_rrule(copy∘broadcasted, BS1, getindex, [rand(3) for _ in 1:2], [3,1], check_inferred=false)
        test_rrule(copy∘broadcasted, BS1, getindex, [rand(3) for _ in 1:2], (3,1), check_inferred=false)
        test_rrule(copy∘broadcasted, BS1, getindex, [rand(3) for _ in 1:2], Ref(CartesianIndex(2)), check_inferred=false)
        test_rrule(copy∘broadcasted, BT1, getindex, Tuple([rand(3) for _ in 1:2]), (3,1), check_inferred=false)
        test_rrule(copy∘broadcasted, BT1, getindex, Tuple([Tuple(rand(3)) for _ in 1:2]), (3,1), check_inferred=false)

        # Protected by Ref/Tuple:
        test_rrule(copy∘broadcasted, BS1, *, rand(3), Ref(rand(2)), check_inferred=false)
        test_rrule(copy∘broadcasted, BS1, conj∘*, rand(3), Ref(rand() + im), check_inferred=false)
        test_rrule(copy∘broadcasted, BS1, conj∘*, rand(3), Ref(rand(2) .+ im), check_inferred=false)
        test_rrule(copy∘broadcasted, BS1, /, (rand(2),), rand(3), check_inferred=false)
    end

    @testset "fused rules" begin
        @testset "arithmetic" begin
            @gpu test_rrule(copy∘broadcasted, +, rand(3), rand(3))
            @gpu test_rrule(copy∘broadcasted, +, rand(3), rand(4)')
            @gpu test_rrule(copy∘broadcasted, +, rand(3), rand(1), rand())
            @gpu test_rrule(copy∘broadcasted, +, rand(3), 1.0*im)
            @gpu test_rrule(copy∘broadcasted, +, rand(3), true)
            @gpu_broken test_rrule(copy∘broadcasted, +, rand(3), Tuple(rand(3)))

            @gpu test_rrule(copy∘broadcasted, -, rand(3), rand(3))
            @gpu test_rrule(copy∘broadcasted, -, rand(3), rand(4)')
            @gpu test_rrule(copy∘broadcasted, -, rand(3))
            test_rrule(copy∘broadcasted, -, Tuple(rand(3)))

            @gpu test_rrule(copy∘broadcasted, *, rand(3), rand(3))
            @gpu test_rrule(copy∘broadcasted, *, rand(3), rand())
            @gpu test_rrule(copy∘broadcasted, *, rand(), rand(3))

            test_rrule(copy∘broadcasted, *, rand(3) .+ im, rand(3) .+ 2im)
            test_rrule(copy∘broadcasted, *, rand(3) .+ im, rand() + 3im)
            test_rrule(copy∘broadcasted, *, rand() + im, rand(3) .+ 4im)

            @test_skip test_rrule(copy∘broadcasted, *, im, rand(3))  # MethodError: no method matching randn(::Random._GLOBAL_RNG, ::Type{Complex{Bool}})
            @test_skip test_rrule(copy∘broadcasted, *, rand(3), im)  # MethodError: no method matching randn(::Random._GLOBAL_RNG, ::Type{Complex{Bool}})
            y4, bk4 = rrule(CFG, copy∘broadcasted, *, im, [1,2,3.0])
            @test y4 == [im, 2im, 3im]
            @test unthunk(bk4([4, 5im, 6+7im])[4]) == [0,5,7]

            # These two test vararg rrule * rule:
            @gpu test_rrule(copy∘broadcasted, *, rand(3), rand(3), rand(3), rand(3), rand(3))
            @gpu_broken test_rrule(copy∘broadcasted, *, rand(), rand(), rand(3), rand(3) .+ im, rand(4)')
            # GPU error from dot(x::JLArray{Float32, 1}, y::JLArray{ComplexF32, 2})

            @gpu test_rrule(copy∘broadcasted, Base.literal_pow, ^, rand(3), Val(2))
            @gpu test_rrule(copy∘broadcasted, Base.literal_pow, ^, rand(3) .+ im, Val(2))

            @gpu test_rrule(copy∘broadcasted, /, rand(3), rand())
            @gpu test_rrule(copy∘broadcasted, /, rand(3) .+ im, rand() + 3im)
        end
        @testset "identity etc" begin
            test_rrule(copy∘broadcasted, identity, rand(3))

            test_rrule(copy∘broadcasted, Float32, rand(3), rtol=1e-4)
            test_rrule(copy∘broadcasted, ComplexF32, rand(3), rtol=1e-4)

            test_rrule(copy∘broadcasted, float, rand(3))
        end
        @testset "complex" begin
            test_rrule(copy∘broadcasted, conj, rand(3))
            test_rrule(copy∘broadcasted, conj, rand(3) .+ im)
            test_rrule(copy∘broadcasted, adjoint, rand(3))
            test_rrule(copy∘broadcasted, adjoint, rand(3) .+ im)

            test_rrule(copy∘broadcasted, real, rand(3))
            test_rrule(copy∘broadcasted, real, rand(3) .+ im)

            test_rrule(copy∘broadcasted, imag, rand(3))
            test_rrule(copy∘broadcasted, imag, rand(3) .+ im .* rand.())

            test_rrule(copy∘broadcasted, complex, rand(3))
        end
    end

    @testset "scalar rules" begin
        @testset "generic" begin
            test_rrule(copy∘broadcasted, BS0, sin, rand())
            test_rrule(copy∘broadcasted, BS0, atan, rand(), rand())
            # test_rrule(copy∘broadcasted, BS0, >, rand(), rand()) # DimensionMismatch from FiniteDifferences
        end
        # Functions with lazy broadcasting rules:
        @testset "arithmetic" begin
            test_rrule(copy∘broadcasted, +, rand(), rand(), rand())
            test_rrule(copy∘broadcasted, +, rand())
            test_rrule(copy∘broadcasted, -, rand(), rand())
            test_rrule(copy∘broadcasted, -, rand())
            test_rrule(copy∘broadcasted, *, rand(), rand())
            test_rrule(copy∘broadcasted, *, rand(), rand(), rand(), rand())
            test_rrule(copy∘broadcasted, Base.literal_pow, ^, rand(), Val(2))
            test_rrule(copy∘broadcasted, /, rand(), rand())
        end
        @testset "identity etc" begin
            test_rrule(copy∘broadcasted, identity, rand())
            test_rrule(copy∘broadcasted, Float32, rand(), rtol=1e-4)
            test_rrule(copy∘broadcasted, float, rand())
        end
        @testset "complex" begin
            test_rrule(copy∘broadcasted, conj, rand())
            test_rrule(copy∘broadcasted, conj, rand() + im)
            test_rrule(copy∘broadcasted, real, rand())
            test_rrule(copy∘broadcasted, real, rand() + im)
            test_rrule(copy∘broadcasted, imag, rand())
            test_rrule(copy∘broadcasted, imag, rand() + im)
            test_rrule(copy∘broadcasted, complex, rand())
        end
    end

    @testset "bugs" begin
        @test ChainRules.unbroadcast((1, 2, [3]), [4, 5, [6]]) isa Tangent   # earlier, NTuple demanded same type
        @test ChainRules.unbroadcast(broadcasted(-, (1, 2), 3), (4, 5)) == (4, 5)  # earlier, called ndims(::Tuple)

        x = Base.Fix1.(*, 1:3.0)
        dx1 = [Tangent{Base.Fix1}(; x = i/2) for i in 1:3, _ in 1:1]
        @test size(ChainRules.unbroadcast(x, dx1)) == size(x)
        dx2 = [Tangent{Base.Fix1}(; x = i/j) for i in 1:3, j in 1:4]
        @test size(ChainRules.unbroadcast(x, dx2)) == size(x)  # was an error, convert(::ZeroTangent, ::Tangent)
        # sum(dx2; dims=2) isa Matrix{Union{ZeroTangent, Tangent{Base.Fix1...}}, ProjectTo copies this so that
        # unbroadcast(x, dx2) isa Vector{Tangent{...}}, that's probably not ideal.

        @test sum(dx2; dims=2)[end] == Tangent{Base.Fix1}(x = 6.25,)
        @test sum(dx1) isa Tangent{Base.Fix1}  # no special code required
    end
end
