struct UltrasphericalCircleWeight{T} <: Weight{T}
    a::T
end

axes(P::UltrasphericalCircleWeight) = (Inclusion(Circle()),)
getindex(P::UltrasphericalCircleWeight, xy::SVector{2}) = xy[2]^P.a


"""
Ortogonal polynomials w.r.t. uniform weight on circle. Equivalent to
Fourier.
"""
struct LegendreCircle{T} <: AlgebraicOrthogonalPolynomial{2,T} end
LegendreCircle() = LegendreCircle{Float64}()

axes(P::LegendreCircle) = (Inclusion(UnitCircle()),_BlockedUnitRange(1:2:∞))

function getindex(P::LegendreCircle, xy::SVector{2}, j::BlockIndex{1})
    x,y = xy
    K,k = block(j),blockindex(j)
    k == 1 ? ChebyshevT()[x,Int(K)] : y*ChebyshevU()[x,Int(K)-1]
end


function getindex(P::LegendreCircle, xy::SVector{2}, j::Int)
    j == 1 && return P[xy, Block(1)[1]]
    P[xy, Block((j ÷ 2)+1)[1+isodd(j)]]
end

"""
Ortogonal polynomials w.r.t. y^a
"""
struct UltrasphericalCircle{T} <: AlgebraicOrthogonalPolynomial{2,T}
    a::T
end

axes(P::UltrasphericalCircle) = (Inclusion(UnitCircle()),_BlockedUnitRange(1:2:∞))

function getindex(P::UltrasphericalCircle, xy::SVector{2}, j::BlockIndex{1})
    x,y = xy
    K,k = block(j),blockindex(j)
    k == 1 ? Jacobi((P.a - 1)/2, (P.a - 1)/2)[x,Int(K)] : y*Jacobi((P.a + 1)/2, (P.a + 1)/2)[x,Int(K)-1]
end

function getindex(P::UltrasphericalCircle, xy::SVector{2}, j::Int)
    j == 1 && return P[xy, Block(1)[1]]
    P[xy, Block((j ÷ 2)+1)[1+isodd(j)]]
end