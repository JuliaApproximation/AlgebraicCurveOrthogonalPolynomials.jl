struct Square{T} <: EuclideanDomain{2,T} end
Square() = Square{Float64}()

function in(p::SVector{2}, d::Square)
    x,y = p
    (abs(x) == 1 && -1 ≤ y ≤ 1) || (abs(y) == 1 && -1 ≤ x ≤ 1)
end

struct SquareLegendre{T} <: OrthogonalPolynomial{T} end
SquareLegendre() = SquareLegendre{Float64}()

axes(P::SquareLegendre) = (Inclusion(Square()),blockedrange([1:3; Fill(4,∞)]))


# function getindex(P::SquareLegendre{T}, xy::SVector{2}, K::Block) where T
#     x,y = xy
#     n = Int(K)-1
#     m = n÷2
#     K == Block(1) && return T[1]
#     K == Block(2) && return T[x, y]
#     K == Block(3) && return T[wedgep(m,x^2,y^2),x*y,wedgeq(2,x^2,y^2)]
#     iseven(n) && return T[wedgep(m,x^2,y^2),wedgeq(m,x^2,y^2),x*y*wedgep(m-1,x^2,y^2),x*y*wedgeq(m-1,x^2,y^2)]
#     T[wedgep(m,x^2,y^2),wedgeq(m,x^2,y^2),x*y*wedgep(m-1,x^2,y^2),x*y*wedgeq(m-1,x^2,y^2)]
# end