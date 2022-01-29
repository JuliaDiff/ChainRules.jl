@testset "UniformScaling rules" begin

    @testset "constructor" begin
        test_rrule(UniformScaling, rand())
    end

    @testset "+" begin
        # Forward
        @test_skip test_frule(+, rand(3,3), I * rand(ComplexF64))  # MethodError: no method matching +(::Matrix{Float64}, ::Tangent{UniformScaling{ComplexF64}, NamedTuple{(:λ,), Tuple{ComplexF64}}})
        test_frule(+, I, rand(3,3))

        # Reverse
        test_rrule(+, rand(3,3), I)
        test_rrule(+, rand(3,3), I * rand(ComplexF64))
        test_rrule(+, I, rand(3,3))
        test_rrule(+, I * rand(), rand(ComplexF64, 3,3))
    end

    @testset "-" begin
        # Forward
        @test_skip test_frule(-, rand(3,3), I * rand(ComplexF64))  # MethodError: no method matching +(::Matrix{Float64}, ::Tangent{UniformScaling{ComplexF64}, NamedTuple{(:λ,), Tuple{ComplexF64}}})
        test_frule(-, I, rand(3,3))

        # Reverse
        test_rrule(-, rand(3,3), I)
        test_rrule(-, rand(3,3), I * rand(ComplexF64))
        test_rrule(-, I, rand(3,3))
        test_rrule(-, I * rand(), rand(ComplexF64, 3,3))
    end

    @testset "Matrix" begin
        test_rrule(Matrix, I, (2, 2))
        test_rrule(Matrix{ComplexF64}, rand()*I, (3, 3))
        test_rrule(Matrix, rand(ComplexF64)*I, (2, 4))
    end

end
