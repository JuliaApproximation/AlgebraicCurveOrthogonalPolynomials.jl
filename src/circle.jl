struct CircleCoordinate{T} <: StaticVector{2,T}
    θ::T
end

getindex(S::CircleCoordinate, k::Int) = k == 1 ? cos(S.θ) : sin(S.θ)
norm(S::CircleCoordinate{T}) where T = one(T)

in(::CircleCoordinate, ::UnitCircle) = true

function convert(::Type{CircleCoordinate{T}}, xy::SVector{2}) where T
    x,y = xy
    CircleCoordinate{T}(atan(y,x))
end

const CircleInclusion{T} = Inclusion{CircleCoordinate{T},UnitCircle{T}}
CircleInclusion{T}() where T = Inclusion{CircleCoordinate{T},UnitCircle{T}}(UnitCircle{T}())

struct UltrasphericalCircleWeight{T} <: Weight{T}
    a::T
end

axes(P::UltrasphericalCircleWeight{T}) where T = (CircleInclusion{T}(),)
getindex(P::UltrasphericalCircleWeight, xy::StaticVector{2}) = xy[2]^P.a


"""
Ortogonal polynomials w.r.t. uniform weight on circle. Equivalent to
Fourier.
"""
struct LegendreCircle{T} <: AlgebraicOrthogonalPolynomial{2,T} end
LegendreCircle() = LegendreCircle{Float64}()



axes(P::LegendreCircle{T}) where T = (CircleInclusion{T}(), _BlockedUnitRange(1:2:∞))

function getindex(P::LegendreCircle{T}, xy::StaticVector{2}, j::BlockIndex{1}) where T
    x,y = xy
    K,k = block(j),blockindex(j)
    K == Block(1) && return one(T)
    k == 1 ? y*ChebyshevU{T}()[x,Int(K)-1] : ChebyshevT{T}()[x,Int(K)]
end


function getindex(P::LegendreCircle, xy::StaticVector{2}, j::Int)
    j == 1 && return P[xy, Block(1)[1]]
    P[xy, Block((j ÷ 2)+1)[1+isodd(j)]]
end

checkpoints(::LegendreCircle{T}) where T = CircleCoordinate.(checkpoints(Fourier{T}()))
grid(Pn::SubQuasiArray{T,2,<:LegendreCircle,<:Tuple{<:Inclusion,<:AbstractUnitRange}}) where T = 
    CircleCoordinate.(grid(Fourier{T}()[:,parentindices(Pn)[2]]))
factorize(Pn::SubQuasiArray{T,2,<:LegendreCircle,<:Tuple{<:Inclusion,<:OneTo}}) where T =
    TransformFactorization(grid(Pn), ShuffledRFFT{T}(size(Pn,2)))
    


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