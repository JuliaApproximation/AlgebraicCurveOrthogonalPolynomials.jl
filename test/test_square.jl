using OrthogonalPolynomialsQuasi, FastGaussQuadrature, StaticArrays, BlockArrays, Test

Y¹ = function(n,x,y)
    if n == 0
        one(x)
    elseif n == 1
        x
    elseif iseven(n)
        wedgep(n÷2,x^2,y^2)
    else # isodd(n)
        x*wedgep(n÷2,x^2,y^2)
    end
end

Y² = function(n,x,y)
    if n == 1
        y
    elseif n == 2
        x*y
    elseif iseven(n)
        wedgeq(n÷2,x^2,y^2)
    else # isodd(n)
        x*wedgeq(n÷2,x^2,y^2)
    end
end

Y³ = function(n,x,y)
    if n == 2
        wedgeq(2,x^2,y^2)
    elseif iseven(n)
        wedgeq(n÷2,x^2,y^2)
    else # isodd(n)
        x*wedgeq(n÷2,x^2,y^2)
    end
end