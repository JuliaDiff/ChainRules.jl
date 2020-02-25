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
            @test isequal(extern(x̄ .+ ∂x), x̄ .+ ȳ .* cos.(x))

            x̄, ȳ = Zero(), rand(3, 3)
            ∂x = pullback(ȳ)[3]
            @test extern(extern(x̄ .+ ∂x)) == ȳ .* cos.(x)
        end
        @testset "frule" begin
            x = rand(3, 3)
            y, ẏ = frule((Zero(), Zero(), One()), broadcast, sin, x)
            @test y == sin.(x)
            @test extern(ẏ) == cos.(x)
        end
    end
end
