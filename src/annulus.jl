annulus(ρ::T) where T = UnitDisk{T}() \ (ρ*UnitDisk{T}())


"""
    AnnulusWeight(a, b)

is a quasi-vector representing `(r^2-ρ)^a * (1-r^2)^b`
"""
struct AnnulusWeight{T} <: Weight{T}
    ρ::T
    a::T
    b::T
end


AnnulusWeight{T}() where T = AnnulusWeight{T}(zero(T), zero(T))
AnnulusWeight() = AnnulusWeight{Float64}()

copy(w::AnnulusWeight) = w

axes(w::AnnulusWeight{T}) where T = (Inclusion(annulus(w.ρ)),)

==(w::AnnulusWeight, v::AnnulusWeight) = w.a == v.a && w.b == v.b && w.r == v.r

function getindex(w::AnnulusWeight, xy::StaticVector{2})
    r = norm(xy)
    (r^2- ρ^2)^w.a * (1-r^2)^w.b
end

"""
    Zernike(a, b)

is a quasi-matrix orthogonal `(r^2 - ρ^2)^a * (1-r^2)^b`.
"""
struct ZernikeAnnulus{T} <: BivariateOrthogonalPolynomial{T}
    ρ::T
    a::T
    b::T
    ZernikeAnnulus{T}(a::T, b::T) where T = new{T}(a, b)
end
ZernikeAnnulus{T}(a, b) where T = ZernikeAnnulus{T}(convert(T,a), convert(T,b))
ZernikeAnnulus(a::T, b::V) where {T,V} = ZernikeAnnulus{float(promote_type(T,V))}(a, b)
ZernikeAnnulus{T}(b) where T = ZernikeAnnulus{T}(zero(b), b)
ZernikeAnnulus{T}() where T = ZernikeAnnulus{T}(zero(T))

ZernikeAnnulus() = ZernikeAnnulus{Float64}()

axes(P::ZernikeAnnulus{T}) where T = (Inclusion(annulus(P.ρ)),blockedrange(oneto(∞)))

==(w::ZernikeAnnulus, v::ZernikeAnnulus) = w.ρ == v.ρ && w.a == v.a && w.b == v.b

copy(A::ZernikeAnnulus) = A

orthogonalityweight(Z::ZernikeAnnulus) = AnnulusWeight(Z.ρ, Z.a, Z.b)

zernikeannulusr(ρ, ℓ, m, a, b, r::T) where T = sqrt(convert(T,2)^(m+a+b+2-iszero(m))/π) * r^m * SemiclassicalJacobi(b,a,m)[(r^2 - 1)/(ρ^2 - 1), (ℓ-m) ÷ 2 + 1]
function zernikeannulusz(ℓ, ms, a, b, rθ::RadialCoordinate{T}) where T
    r,θ = rθ.r,rθ.θ
    m = abs(ms)
    zernikeannulusr(ℓ, m, a, b, r) * (signbit(ms) ? sin(m*θ) : cos(m*θ))
end

zernikeannulusz(ℓ, ms, a, b, xy::StaticVector{2}) = zernikeannulusz(ℓ, ms, a, b, RadialCoordinate(xy))
zernikeannulusz(ℓ, ms, b, xy::StaticVector{2}) = zernikeannulusz(ℓ, ms, zero(b), b, xy)
zernikeannulusz(ℓ, ms, xy::StaticVector{2,T}) where T = zernikeannulusz(ℓ, ms, zero(T), xy)

function getindex(Z::ZernikeAnnulus{T}, rθ::RadialCoordinate, B::BlockIndex{1}) where T
    ℓ = Int(block(B))-1
    k = blockindex(B)
    m = iseven(ℓ) ? k-isodd(k) : k-iseven(k)
    zernikeannulusz(ℓ, (isodd(k+ℓ) ? 1 : -1) * m, Z.a, Z.b, rθ)
end


getindex(Z::ZernikeAnnulus, xy::StaticVector{2}, B::BlockIndex{1}) = Z[RadialCoordinate(xy), B]
getindex(Z::ZernikeAnnulus, xy::StaticVector{2}, B::Block{1}) = [Z[xy, B[j]] for j=1:Int(B)]
getindex(Z::ZernikeAnnulus, xy::StaticVector{2}, JR::BlockOneTo) = mortar([Z[xy,Block(J)] for J = 1:Int(JR[end])])



# function \(A::ZernikeAnnulus{T}, B::ZernikeAnnulus{V}) where {T,V}
#     TV = promote_type(T,V)
#     A.a == B.a && A.b == B.b && return Eye{TV}(∞)
#     @assert A.a == 0 && A.b == 1
#     @assert B.a == 0 && B.b == 0
#     ModalInterlace{TV}((Normalized.(Jacobi{TV}.(1,0:∞)) .\ Normalized.(Jacobi{TV}.(0,0:∞))) ./ sqrt(convert(TV, 2)), (0,2))
# end

# function \(A::ZernikeAnnulus{T}, B::Weighted{V,ZernikeAnnulus{V}}) where {T,V}
#     TV = promote_type(T,V)
#     A.a == B.P.a == A.b == B.P.b == 0 && return Eye{TV}(∞)
#     @assert A.a == A.b == 0
#     @assert B.P.a == 0 && B.P.b == 1
#     ModalInterlace{TV}((Normalized.(Jacobi{TV}.(0, 0:∞)) .\ HalfWeighted{:a}.(Normalized.(Jacobi{TV}.(1, 0:∞)))) ./ sqrt(convert(TV, 2)), (2,0))
# end