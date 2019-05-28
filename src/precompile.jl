# Make sure that IRTools has precompiled
function __init__()
    f1, dX1 = rrule(sum, ones(4, 4));
    dX1(f1)
end
