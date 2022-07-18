function rrule(::Type{T}, ps::Pair...) where {T<:Dict}
    @show ps
    ks = map(first, ps)
    project_ks, project_vs = map(ProjectTo, ks), map(ProjectTo∘last, ps)
    function Dict_pullback(ȳ)
        @show ȳ
        dps = map(ks, project_ks, project_vs) do k, proj_k, proj_v
            dk, dv = proj_k(getkey(ȳ, k, NoTangent())), proj_v(get(ȳ, k, NoTangent()))
            Tangent{Pair{typeof(dk), typeof(dv)}}(first = dk, second = dv)
        end
        @show dps
        return (NoTangent(), dps...)
    end
    return T(ps...), Dict_pullback
end