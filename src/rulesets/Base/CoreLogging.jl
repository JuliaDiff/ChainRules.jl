# For the CoreLogging submodule of Base. (not to be confused with the Logging stdlib)

function rrule(
    rc::RuleConfig{>:ChainRulesCore.HasReverseMode},
    ::typeof(Base.CoreLogging.with_logger),
    f::Function,
    logger::Base.CoreLogging.AbstractLogger
)
    y, f_pb = Base.CoreLogging.with_logger(logger) do
        rrule_via_ad(rc, f)
    end
    with_logger_pullback(ȳ) = (NoTangent(), only(f_pb(ȳ)), NoTangent())
    return y, with_logger_pullback
end

@non_differentiable Base.CoreLogging.current_logger(args...)
@non_differentiable Base.CoreLogging.current_logger_for_env(::Any...)
@non_differentiable Base.CoreLogging._invoked_shouldlog(::Any...)
@non_differentiable Base.CoreLogging.Base.fixup_stdlib_path(::Any)
@non_differentiable Base.CoreLogging.handle_message(::Any...)