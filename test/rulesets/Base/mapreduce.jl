@testset "Maps and Reductions" begin
    @testset "sum" begin
        @testset "Vector" begin
            M = 3
            frule_test(sum, (randn(M), randn(M)))
            rrule_test(sum, randn(), (randn(M), randn(M)))
        end
        @testset "Matrix" begin
            M, N = 3, 4
            frule_test(sum, (randn(M, N), randn(M, N)))
            rrule_test(sum, randn(), (randn(M, N), randn(M, N)))
        end
        @testset "Array{T, 3}" begin
            M, N, P = 3, 7, 11
            frule_test(sum, (randn(M, N, P), randn(M, N, P)))
            rrule_test(sum, randn(), (randn(M, N, P), randn(M, N, P)))
        end
        @testset "keyword arguments" begin
            n = 4
            X = randn(n, n+1)
            y, pullback = rrule(sum, X; dims=2)
            ȳ = randn(size(y))
            _, x̄_ad = pullback(ȳ)
            x̄_fd = only(j′vp(central_fdm(5, 1), x->sum(x, dims=2), ȳ, X))
            @test x̄_ad ≈ x̄_fd atol=1e-9 rtol=1e-9
        end
    end  # sum
end
