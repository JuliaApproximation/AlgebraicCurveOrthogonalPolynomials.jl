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
    (r^2- w.ρ^2)^w.a * (1-r^2)^w.b
end


abstract type AbstractZernikeAnnulus{T} <: BivariateOrthogonalPolynomial{T} end
"""
    ZernikeAnnulus(ρ, a, b)

is a quasi-matrix orthogonal `(r^2 - ρ^2)^a * (1-r^2)^b`.
"""
struct ZernikeAnnulus{T} <: AbstractZernikeAnnulus{T}
    ρ::T
    a::T
    b::T
    ZernikeAnnulus{T}(ρ::T, a::T, b::T) where T = new{T}(ρ, a, b)
end

"""
    ComplexZernikeAnnulus(ρ, a, b)

is a complex-valued quasi-matrix orthogonal `(r^2 - ρ^2)^a * (1-r^2)^b`.
"""
struct ComplexZernikeAnnulus{T} <: AbstractZernikeAnnulus{Complex{T}}
    ρ::T
    a::T
    b::T
    ComplexZernikeAnnulus{T}(ρ::T, a::T, b::T) where T = new{T}(ρ, a, b)
end


for Typ in (:ZernikeAnnulus, :ComplexZernikeAnnulus)
    @eval begin
        $Typ{T}(ρ, a, b) where T = $Typ{T}(convert(T,ρ), convert(T,a), convert(T,b))
        $Typ(ρ::R, a::T, b::V) where {R,T,V} = $Typ{float(promote_type(R,T,V))}(ρ, a, b)
        $Typ{T}(ρ) where T = $Typ{T}(ρ, zero(ρ), zero(ρ))
        $Typ(ρ) = $Typ(ρ, zero(ρ), zero(ρ))

        ==(w::$Typ, v::$Typ) = w.ρ == v.ρ && w.a == v.a && w.b == v.b
    end
end



axes(P::AbstractZernikeAnnulus{T}) where T = (Inclusion(annulus(P.ρ)),blockedrange(oneto(∞)))
copy(A::AbstractZernikeAnnulus) = A

orthogonalityweight(Z::AbstractZernikeAnnulus) = AnnulusWeight(Z.ρ, Z.a, Z.b)

zernikeannulusr(ρ, ℓ, m, a, b, r::T) where T = r^m * SemiclassicalJacobi{T}(inv(1-ρ^2),b,a,m)[(r^2 - 1)/(ρ^2 - 1), (ℓ-m) ÷ 2 + 1]
function zernikeannulusz(ρ, ℓ, ms, a, b, rθ::RadialCoordinate{T}) where T
    r,θ = rθ.r,rθ.θ
    m = abs(ms)
    zernikeannulusr(ρ, ℓ, m, a, b, r) * (signbit(ms) ? sin(m*θ) : cos(m*θ))
end

function complexzernikeannulusz(ρ, ℓ, ms, a, b, rθ::RadialCoordinate{T}) where T
    r,θ = rθ.r,rθ.θ
    m = abs(ms)
    zernikeannulusr(ρ, ℓ, m, a, b, r) * exp(im*m*θ)
end


for func in (:zernikeannulusz, :complexzernikeannulusz)
    @eval begin
        $func(ρ, ℓ, ms, a, b, xy::StaticVector{2}) = $func(ρ, ℓ, ms, a, b, RadialCoordinate(xy))
        $func(ρ, ℓ, ms, b, xy::StaticVector{2}) = $func(ρ, ℓ, ms, zero(b), b, xy)
        $func(ρ, ℓ, ms, xy::StaticVector{2,T}) where T = $func(ρ, ℓ, ms, zero(T), xy)
    end
end

function getindex(Z::ZernikeAnnulus{T}, rθ::RadialCoordinate, B::BlockIndex{1}) where T
    ℓ = Int(block(B))-1
    k = blockindex(B)
    m = iseven(ℓ) ? k-isodd(k) : k-iseven(k)
    zernikeannulusz(Z.ρ, ℓ, (isodd(k+ℓ) ? 1 : -1) * m, Z.a, Z.b, rθ)
end


function getindex(Z::ComplexZernikeAnnulus{T}, rθ::RadialCoordinate, B::BlockIndex{1}) where T
    ℓ = Int(block(B))-1
    k = blockindex(B)
    m = iseven(ℓ) ? k-isodd(k) : k-iseven(k)
    complexzernikeannulusz(Z.ρ, ℓ, (isodd(k+ℓ) ? 1 : -1) * m, Z.a, Z.b, rθ)
end


getindex(Z::AbstractZernikeAnnulus, xy::StaticVector{2}, B::BlockIndex{1}) = Z[RadialCoordinate(xy), B]
getindex(Z::AbstractZernikeAnnulus, xy::StaticVector{2}, B::Block{1}) = [Z[xy, B[j]] for j=1:Int(B)]
getindex(Z::AbstractZernikeAnnulus, xy::StaticVector{2}, JR::BlockOneTo) = mortar([Z[xy,Block(J)] for J = 1:Int(JR[end])])



function \(A::AbstractZernikeAnnulus{T}, B::AbstractZernikeAnnulus{V}) where {T,V}
    TV = promote_type(T,V)
    A.a == B.a && A.b == B.b && return Eye{TV}(∞)
    t = inv(1-A.ρ^2)
    ModalInterlace{TV}((SemiclassicalJacobi{TV}.(t,A.b,A.a,0:∞) .\ SemiclassicalJacobi{TV}.(t,B.b,B.a,0:∞)), (ℵ₀,ℵ₀), (0,2*Int(max(A.b-B.b,A.a-B.a))))
end

# function \(A::ZernikeAnnulus{T}, B::Weighted{V,ZernikeAnnulus{V}}) where {T,V}
#     TV = promote_type(T,V)
#     A.a == B.P.a == A.b == B.P.b == 0 && return Eye{TV}(∞)
#     @assert A.a == A.b == 0
#     @assert B.P.a == 0 && B.P.b == 1
#     ModalInterlace{TV}((Normalized.(Jacobi{TV}.(0, 0:∞)) .\ HalfWeighted{:a}.(Normalized.(Jacobi{TV}.(1, 0:∞)))) ./ sqrt(convert(TV, 2)), (2,0))
# end


###
# Laplacian
###


@simplify function *(Δ::Laplacian, W::Weighted{<:Any,<:ZernikeAnnulus})
    P = W.P
    @assert P.a == P.b == 1
    ρ = P.ρ; t = inv(1-ρ^2)
    T = eltype(P)
    Ps = SemiclassicalJacobi{T}.(t,1,1,0:∞)
    D = Derivative(axes(Ps[1],1))
    Δs = BroadcastVector{AbstractMatrix{T}}((C,B,A) -> 4t*(1-ρ^2)^2*divmul(HalfWeighted{:c}(C),D,HalfWeighted{:c}(B))*divmul(HalfWeighted{:ab}(B),D,HalfWeighted{:ab}(A)), Ps, SemiclassicalJacobi.(t,0,0,1:∞), Ps)
    P * ModalInterlace(Δs, (ℵ₀,ℵ₀), (2,2))
end

@simplify function *(Δ::Laplacian, P::ZernikeAnnulus)
    ρ,a,b = P.ρ,P.a,P.b
    t = inv(1-ρ^2)
    T = eltype(P)
    Ps = SemiclassicalJacobi.(t,b,a,0:∞)
    D = Derivative(axes(Ps[1],1))
    Δs = BroadcastVector{AbstractMatrix{T}}((C,B,A) -> 4t*divmul(HalfWeighted{:c}(C),D,HalfWeighted{:c}(B))*divmul(B,D,A), SemiclassicalJacobi.(t,b+2,a+2,0:∞), SemiclassicalJacobi.(t,b+1,a+1,1:∞), Ps)
    ZernikeAnnulus(ρ,a+2,b+2) * ModalInterlace(Δs, (ℵ₀,ℵ₀), (-2,6))
end