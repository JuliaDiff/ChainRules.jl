@testset "Maps and Reductions" begin
    @testset "sum" begin
        sizes = (3, 4, 7)
        @testset "dims = $dims" for dims in (:, 1)
            fkwargs = (dims=dims,)
            @testset "Array{$N, $T}" for N in eachindex(sizes), T in (Float64, ComplexF64)
                s = sizes[1:N]
                x, ẋ, x̄ = randn(T, s...), randn(T, s...), randn(T, s...)
                y = sum(x; dims=dims)
                Δy = randn(eltype(y), size(y)...)
                frule_test(sum, (x, ẋ); fkwargs=fkwargs)
                rrule_test(sum, Δy, (x, x̄); fkwargs=fkwargs)
            end
        end
    end  # sum

    @testset "sum abs2" begin
        sizes = (3, 4, 7)
        @testset "dims = $dims" for dims in (:, 1)
            fkwargs = (dims=dims,)
            @testset "Array{$N, $T}" for N in eachindex(sizes), T in (Float64, ComplexF64)
                s = sizes[1:N]
                x, ẋ, x̄ = randn(T, s...), randn(T, s...), randn(T, s...)
                y = sum(abs2, x; dims=dims)
                Δy = randn(eltype(y), size(y)...)
                @testset "frule" begin
                    # can't use frule_test here because it doesn't yet ignore nothing tangents
                    y_ad, ẏ_ad = frule((Zero(), Zero(), ẋ), sum, abs2, x; dims=dims)
                    @test y_ad == y
                    ẏ_fd = jvp(_fdm, z -> sum(abs2, z; dims=dims), (x, ẋ))
                    @test ẏ_ad ≈ ẏ_fd
                end
                @testset "rrule" begin
                    rrule_test(sum, Δy, (abs2, nothing), (x, x̄); fkwargs=fkwargs)
                end
            end
        end
    end  # sum abs2
end
