"""
    parameters(type)
Extracts the type-parameters of the `type`.
e.g. `parameters(Foo{A, B, C}) == [A, B, C]`
"""
parameters(sig::UnionAll) = parameters(sig.body)
parameters(sig::DataType) = sig.parameters
parameters(sig::Union) = Base.uniontypes(sig)

@testset "Make sure methods haven't been added to DataType/UnionAll/Union" begin
    # if someone wrote e.g. `rrule(::typeof(Foo), x)` rather than `rrule(::Type{<:Foo}, x)`
    # then that would actually define `rrule(::DataType, x)` which would be bad
    # This test checks for that and fails if such a method exists.
    for method in methods(rrule)
        function_type = if method.sig <: Tuple{typeof(rrule), RuleConfig, Type, Vararg}
            parameters(method.sig)[3]
        elseif method.sig <: Tuple{typeof(rrule), Type, Vararg}
            parameters(method.sig)[2]
        else
            nothing
        end
        
        if function_type == DataType || function_type == UnionAll || function_type == Union
            @error "Bad constructor rrule. typeof(T)` not `Type{T}`" method
            @test false
        end
    end

    # frule
    for method in methods(frule)
        function_type = if method.sig <: Tuple{typeof(frule), RuleConfig, Any, Type, Vararg}
            parameters(method.sig)[4]
        elseif method.sig <: Tuple{typeof(frule), Any, Type, Vararg}
            @show parameters(method.sig)[3]
        else
            nothing
        end
        
        if function_type == DataType || function_type == UnionAll || function_type == Union
            @error "Bad constructor frule. typeof(T)` not `Type{T}`" method
            @test false
        end
    end
end
