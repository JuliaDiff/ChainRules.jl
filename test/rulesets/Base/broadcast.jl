@testset "broadcast" begin
    @testset "sin.(x)" begin
        @testset "rrule" begin
            x = rand(3, 3)
            y, pullback = rrule(broadcast, sin, x)
            @test y == sin.(x)
            (dself, dsin, dx) = pullback(One())
            @test dself == NO_FIELDS
            @test dsin == DoesNotExist()
            @test extern(dx) == cos.(x)

            x̄, ȳ = rand(), rand()
            ∂x = pullback(ȳ)[3]
            @test isequal(
                extern(ChainRules.accumulate(x̄, ∂x)),
                x̄ .+ ȳ .* cos.(x)
            )

            x̄, ȳ = Zero(), rand(3, 3)
            ∂x = pullback(ȳ)[3]
            @test extern(extern(accumulate(x̄, ∂x))) == ȳ .* cos.(x)
        end
        @testset "frule" begin
            x = rand(3, 3)
            y, pushforward = frule(broadcast, sin, x)
            @test y == sin.(x)

            ẏ = pushforward(NamedTuple(), NamedTuple(), One())
            @test extern(ẏ) == cos.(x)
        end
    end
end
