@testset "Differentials" begin
    @testset "Wirtinger" begin
        w = Wirtinger(1+1im, 2+2im)
        @test wirtinger_primal(w) == 1+1im
        @test wirtinger_conjugate(w) == 2+2im
        @test +(w, w) == Wirtinger(2+2im, 4+4im)
        # TODO: other add_wirtinger methods stack overflow
        @test_throws ErrorException (w * w)
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
        @test +(z, z) == z
        @test +(z, 1) == 1
        @test +(1, z) == 1
        @test *(z, z) == z
        @test *(z, 1) == z
        @test *(1, z) == z
        for x in z
            @test x === z
        end
        @test broadcastable(z) isa Ref{Zero}
        @test conj(z) == z
    end
    @testset "One" begin
        o = One()
        @test extern(o) === true
        @test +(o, o) == 2
        @test +(o, 1) == 2
        @test +(1, o) == 2
        @test *(o, o) == o
        @test *(o, 1) == 1
        @test *(1, o) == 1
        for x in o
            @test x === o
        end
        @test broadcastable(o) isa Ref{One}
        @test conj(o) == o
    end
end
