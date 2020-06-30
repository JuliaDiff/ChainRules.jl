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
        @testset for T in (Float64, ComplexF64)
            @testset "Vector" begin
                M = 3
                rrule_test(sum, randn(), (abs2, nothing), (randn(T, M), randn(T, M)))
            end
            @testset "Matrix" begin
                M, N = 3, 4
                rrule_test(sum, randn(), (abs2, nothing), (randn(T, M, N), randn(T, M, N)))
            end
            @testset "Array{T, 3}" begin
                M, N, P = 3, 7, 11
                rrule_test(sum, randn(), (abs2, nothing), (randn(T, M, N, P), randn(T, M, N, P)))
            end
            @testset "keyword arguments" begin
                n = 4
                rrule_test(
                    sum,
                    randn(n, 1),
                    (abs2, nothing),
                    (randn(T, n, n+1), randn(T, n, n+1));
                    fkwargs=(dims=2,),
                )
            end
        end
    end  # sum abs2
end
