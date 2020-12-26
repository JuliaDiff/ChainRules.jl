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
                test_rrule(sum, abs2 ⊢ DoesNotExist(), x; fkwargs=(;dims=dims))
            end
        end
    end  # sum abs2

    @testset "prod" begin
        @testset "Array{$T}" for T in [Float64, ComplexF64]
            @testset "size = $sz, dims = $dims" for (sz, dims) in [
                ((12,), :), ((12,), 1),
                ((3,4), 1), ((3,4), 2), ((3,4), :), ((3,4), [1,2]),
                ((3,4,1), 1), ((3,2,2), 3), ((3,2,2), 2:3),
                ]
                x, xdot, xbar = randn(T, sz), randn(T, sz), randn(T, sz)
                # frule_test(prod, (x, xdot); fkwargs=(dims=dims,))
                rrule_test(prod, prod(x; dims=dims), (x, xbar); fkwargs=(dims=dims,))

                x[1] = 0
                rrule_test(prod, prod(x; dims=dims), (x, xbar); fkwargs=(dims=dims,))

                x[5] = 0
                rrule_test(prod, prod(x; dims=dims), (x, xbar); fkwargs=(dims=dims,))

                x[3] = x[7] = 0  # two zeros along some slice, for any dims
                rrule_test(prod, prod(x; dims=dims), (x, xbar); fkwargs=(dims=dims,))

                if ndims(x) == 3
                    xp = PermutedDimsArray(x, (3,2,1))  # not a StridedArray
                    xpdot, xpbar = permutedims(xdot, (3,2,1)), permutedims(xbar, (3,2,1))
                    # frule_test(prod, (xp, xpdot); fkwargs=dims)
                    rrule_test(prod, prod(xp; dims=dims), (xp, xpbar); fkwargs=(dims=dims,))
                end
            end
        end
        @testset "Array{Float32}" begin
            v = [1f-5, 1f-10, 1f-15, 1f-20]
            @test prod(v) == 0
            vbar = randn(Float32, 4)
            @test unthunk(rrule(prod, v)[2](1f0)[2]) == zeros(4)
            rrule_test(prod, 0f0, (v, vbar))
        end
    end # prod
end
