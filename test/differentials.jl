@testset "Differentials" begin
    @testset "Wirtinger" begin
        w = Wirtinger(1+1im, 2+2im)
        @test wirtinger_primal(w) == 1+1im
        @test wirtinger_conjugate(w) == 2+2im
        @test add_wirtinger(w, w) == Wirtinger(2+2im, 4+4im)
        # TODO: other add_wirtinger methods stack overflow
        @test_throws ErrorException mul_wirtinger(w, w)
        @test_throws ErrorException extern(w)
        for x in w
            @test x === w
        end
        @test broadcastable(w) == w
        @test_throws ErrorException conj(w)
    end
    @testset "Zero" begin
        z = Zero()
        @test extern(z) === false
        @test add_zero(z, z) == z
        @test add_zero(z, 1) == 1
        @test add_zero(1, z) == 1
        @test mul_zero(z, z) == z
        @test mul_zero(z, 1) == z
        @test mul_zero(1, z) == z
        for x in z
            @test x === z
        end
        @test broadcastable(z) isa Ref{Zero}
        @test conj(z) == z
    end
    @testset "One" begin
        o = One()
        @test extern(o) === true
        @test add_one(o, o) == 2
        @test add_one(o, 1) == 2
        @test add_one(1, o) == 2
        @test mul_one(o, o) == o
        @test mul_one(o, 1) == 1
        @test mul_one(1, o) == 1
        for x in o
            @test x === o
        end
        @test broadcastable(o) isa Ref{One}
        @test conj(o) == o
    end
end
