
@testset "sparse(I, J, V, m, n, +)" begin
    m, n = 3, 5
    s, t, w = [1,2], [2,3], [0.5,0.5]
    
    test_rrule(sparse, s, t, w, m, n, +)
end
