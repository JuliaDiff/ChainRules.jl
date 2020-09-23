try:
    using Zygote: @adjoint

    ignore(f) = f()
    @adjoint ignore(f) = ignore(f), _ -> nothing

    macro ignore(ex)
        return :(ChainRules.ignore() do
            $(esc(ex))
        end)
    end
catch
    nothing