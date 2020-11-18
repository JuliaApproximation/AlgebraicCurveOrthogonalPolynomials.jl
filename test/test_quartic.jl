using OrthogonalPolynomialsAlgebraicCurves, BandedMatrices, BlockBandedMatrices, BlockArrays, Test
using ForwardDiff, StaticArrays

@testset "x^2 + y^4 = 1" begin
    P = UltrasphericalArc()
    Q = UltrasphericalArc(2, P)
    xy² = axes(P,1)
    x = first.(xy²)
    y² = last.(xy²)

    # 	P_00(x,y^2), 
	# P_10(x,y^2),	y*Q_00(x,y^2), 
	# P_20(x,y^2), 	y*Q_10(x,y^2),	P_11(x,y^2)
	# P_30(x,y^2), 	y*Q_20(x,y^2),	y*Q_11(x,y^2),		P_21(x,y^2)
	# P_40(x,y^2), 	y*Q_30(x,y^2),	y*Q_21(x,y^2),		P_31(x,y^2)
    # …
    
    Q \ P
end


function F_quartic(x)
    Ax,Bx,Ay,By = unroll(x)
    N = 9
    n = size(Ax,1)
    X = blocksymtricirculant(Ax, Bx, N)
    Y = blocksymtricirculant(Ay, By, N)
    II = Matrix(I,n,n)
    ZZ = zeros(n,n)
    E = mortar([ZZ, II, Fill(ZZ,N-2)...]')'
    Σ = mortar(Fill(II, N)') # evaluate at 1
    Ω = mortar(((-1).^(1:N) .* Fill(II, N))') # evaluate at -1

    vcat(map(vec,[Σ*X*E - I, Σ*Y*E, Ω*X*E + I, Ω*Y*E ,  X*Y - Y*X, X^4 + Y^4 - I])...)
    # vcat(map(vec,[X*Y - Y*X, X^4 + Y^4 - I])...)
end

@testset "quartic Newton" begin
    n = 32
    p = randn(2(sum(1:n) + n^2))
    for _=1:10
        global p
        J = ForwardDiff.jacobian(F_quartic,p); p = p - (J \ F_quartic(p)); 
        @show F_quartic(p) |> norm
    end; p
    Ax,Bx,Ay,By = unroll(p)
    x = z -> Ax + Bx/z + z*Bx'
    y = z -> Ay + By/z + z*By'
    λ,Q = eigen(Hermitian(x(z)))
    Q'y(z)*Q
end

z = exp(0.1im)
Bx/z + z*Bx'
using Plots
scatter(real(z) .+ im*eigvals(Hermitian(By/z + z*By')))

NN = 20; Z = Matrix{ComplexF64}(undef,size(Ax,1),NN)
for (j,θ) in enumerate(range(0,2π; length=NN))
    z = exp(θ*im)
    λ,Q = eigen(x(z))
    Z[:,j] = λ  .+ im*real(diag(Q'*y(z)*Q))
end

@test real(Z).^4 + imag(Z).^4 ≈ Ones(4,NN)

scatter(real(Z), imag(Z))

By

function F_quartic(x)
    n = 4
    Bx = reshape(x,n,2n)[:,1:n]
    By = reshape(x,n,2n)[:,n+1:end]
    N = 9
    ZZ = zeros(eltype(x),n,n)
    X = blocksymtricirculant(ZZ, Bx, N)
    Y = blocksymtricirculant(ZZ, By, N)

    # vcat(map(vec,[Σ*X*E - I, Ω*X*E + I,  X*Y - Y*X, X^4 + Y^4 - I])...)
    vcat(map(vec,[X*Y - Y*X, X^4 + Y^4 - I])...)
end


p = randn(32)
J = ForwardDiff.jacobian(F_quartic,p); p = p - (J \ F_quartic(p)); norm(F_quartic(p))
n = 4
Bx = reshape(p,n,2n)[:,1:n]
By = reshape(p,n,2n)[:,n+1:end]

F_quartic(p)

