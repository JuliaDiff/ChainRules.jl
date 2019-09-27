using Pkg: @pkg_str
# For reproducability only dependency this has is Luxor,
# and it was created with Luxor v1.6.0
pkg"add Luxor@v1.6"

using Luxor
using Random

const bridge_len = 50

function chain(jiggle=0)
    shaky_rotate(θ) = rotate(θ + jiggle*(rand()-0.5))
    
    ### 1
    shaky_rotate(0)
    sethue(Luxor.julia_red)
    link()
    m1 = getmatrix()
    
    
    ### 2
    sethue(Luxor.julia_green)
    translate(-50, 130);
    shaky_rotate(π/3); 
    link()
    m2 = getmatrix()
    
    setmatrix(m1)
    sethue(Luxor.julia_red)
    overlap(-1.3π)
    setmatrix(m2)
    
    ### 3
    shaky_rotate(-π/3);
    translate(-120,80);
    sethue(Luxor.julia_purple)
    link()
    
    setmatrix(m2)
    setcolor(Luxor.julia_green)
    overlap(-1.5π)
end


function link()
    sector(50, 90, π, 0, :fill)
    sector(Point(0, bridge_len), 50, 90, 0, -π, :fill)
    
    
    rect(50,-3,40, bridge_len+6, :fill)
    rect(-50-40,-3,40, bridge_len+6, :fill)
    
    sethue("black")
    move(Point(-50, bridge_len))
    arc(Point(0,0), 50, π, 0, :stoke)
    arc(Point(0, bridge_len), 50, 0, -π, :stroke)
    
    move(Point(-90, bridge_len))
    arc(Point(0,0), 90, π, 0, :stoke)
    arc(Point(0, bridge_len), 90, 0, -π, :stroke)
    strokepath()
end

function overlap(ang_end)
    sector(Point(0, bridge_len), 50, 90, -0., ang_end, :fill)
    sethue("black")
    arc(Point(0, bridge_len), 50, 0, ang_end, :stoke)
    move(Point(90, bridge_len))
    arc(Point(0, bridge_len), 90, 0, ang_end, :stoke)

    strokepath()
end

####################
# Actually draw it

Random.seed!(16)
Drawing(450,450, "logo.svg")
origin()
translate(50, -130);
chain(0.5)
finish()
preview()