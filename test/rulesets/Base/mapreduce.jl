@testset "Maps and Reductions" begin
    @testset "sum" begin
        sizes = (3, 4, 7)
        @testset "dims = $dims" for dims in (:, 1)
            @testset "Array{$N, $T}" for N in eachindex(sizes), T in (Float64, ComplexF64)
                x = randn(T, sizes[1:N]...)
                test_frule(sum, x; fkwargs=(;dims=dims))
                test_rrule(sum, x; fkwargs=(;dims=dims))
            end
        end
    end  # sum

    @testset "sum abs2" begin
        sizes = (3, 4, 7)
        @testset "dims = $dims" for dims in (:, 1)
            @testset "Array{$N, $T}" for N in eachindex(sizes), T in (Float64, ComplexF64)
                x = randn(T, sizes[1:N]...)
                test_frule(sum, abs2, x; fkwargs=(;dims=dims))
                test_rrule(sum, abs2 ⊢ nothing, x; fkwargs=(;dims=dims))
            end
        end
    end  # sum abs2

    @testset "prod" begin
        test_rrule(prod, randn(5))
        test_rrule(prod, randn(5, 6))
        test_rrule(prod, randn(5, 6); fkwargs=(;dims=2))
        test_rrule(prod, randn(5, 6); fkwargs=(;dims=1))
    end
end
