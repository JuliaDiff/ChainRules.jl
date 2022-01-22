
@testset "sparse(I, J, V)" begin
    m, n = 3, 5
    s, t, w = [1,2], [2,3], [0.5,0.5]
    
    g = gradient(w -> sum(sparse(s, t, w, m, n)), w)
    # @show g
    test_rrule(sparse, s, t, w, m, n)
end
