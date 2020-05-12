using OrthogonalPolynomialsAlgebraicCurves, BandedMatrices, BlockBandedMatrices, BlockArrays
using ForwardDiff, StaticArrays



@testset "Circulant" begin    
    Ax = Matrix(0.5I,2,2)
    Bx = Matrix(0.25I,2,2)
    a₁₂ = (1 + sqrt(2))/4
    a₂₁ = (1 - sqrt(2))/4
    Ay = [0 -0.5; -0.5 0]
    By = [0 a₁₂; a₂₁ 0]

    N = 5
    X = blocksymtricirculant(Ax, Bx, N)
    Y = blocksymtricirculant(Ay, By, N)
    @test X == X'
    @test Y == Y'
    @test X*Y ≈ Y*X
    @test X^2 + Y^2 ≈ I(2N)

    x = z -> Ax + (Bx/z + Bx'*z)
    y = z -> Ay + (By/z + By'*z)
    @test x(1) ≈ I(2)
    @test norm(y(1)) ≤ 10eps()
    @test norm(x(-1)) ≤ 10eps()
    @test y(-1) ≈ [0 -1; -1 0]
    @test eigvals(y(-1)) ≈ [-1,1]

    @testset "Nonlinear Function" begin
        E = mortar([zeros(2,2), [1 0; 0 1], Fill(zeros(2,2),N-2)...]')'
        Σ = mortar(Fill([1 0; 0 1], N)')
        Ω = mortar(((-1).^(1:N) .* Fill([1 0; 0 1], N))')
        @test Σ*X*E ≈ Ax + Bx + Bx'
        @test Ω*X*E ≈ Ax - Bx - Bx'

        F = (X,Y) -> [Σ*X*E - I, Σ*Y*E, Ω*X*E, Ω*Y*E - [0 -1; -1 0],  X*Y - Y*X, X^2 + Y^2 - I]
        @test norm(norm.(F(X,Y))) ≤ 10eps()
    end


    # @testset "Nonlinear solve" begin
    #     xd = randn(7)
    #     yd = randn(7)
    #     Ax,Bx = unroll(xd...)
    #     Ay,By = unroll(yd...)
    #     N = 7
    #     X = blocksymtricirculant(Ax, Bx, N)
    #     Y = blocksymtricirculant(Ay, By, N)

    #     [(X*Y - Y*X)[2:5,1]; (X*Y - Y*X)[3,2]; (X*Y - Y*X)[5,2]; (X^2 + Y^2 - I)[1:6,1]; (X^2 + Y^2 - I)[2:6,2]]

    #     Ax*Ay-Ay*Ax + Bx*By'-By'*Bx + Bx'*By - By*Bx'
    #     Ax*By-Ay*Bx + Bx*Ay-By*Ax
    #     Bx*By - By*Bx

    #     Ax^2 + Bx*Bx' + Bx'*Bx + Ay^2 + By*By' + By'*By - I
    #     Bx*Ax + Ax*Bx + By*Ay + Ay*By # Bx'*Ax + Ax*Bx' + By'*Ay + Ay*By'
    #     Bx^2 + By^2
    # end
end


# Ax = [0 0.0; 0.0 0]; Bx = [0.5 0; 0 0.5]
# Ay = [0 0.0; 0.0 0]; By = [0 0.5; -0.5 0]
function F_circle(x)
    Ax,Bx,Ay,By = unroll(x)
    N = 5
    X = blocksymtricirculant(Ax, Bx, N)
    Y = blocksymtricirculant(Ay, By, N)
    E = mortar([zeros(2,2), [1 0; 0 1], Fill(zeros(2,2),N-2)...]')'
    Σ = mortar(Fill([1 0; 0 1], N)')
    Ω = mortar(((-1).^(1:N) .* Fill([1 0; 0 1], N))')

    vcat(map(vec,[Σ*X*E - I, Σ*Y*E, Ω*X*E + I, Ω*Y*E,  X*Y - Y*X, X^2 + Y^2 - I])...)
    # vcat(map(vec,[Σ*X*E - I, Ω*X*E + I,  X*Y - Y*X, X^2 + Y^2 - I])...)
end

function F_semicircle(x)
    Ax,Bx,Ay,By = unroll(x)
    N = 5
    X = blocksymtricirculant(Ax, Bx, N)
    Y = blocksymtricirculant(Ay, By, N)
    E = mortar([zeros(2,2), [1 0; 0 1], Fill(zeros(2,2),N-2)...]')'
    Σ = mortar(Fill([1 0; 0 1], N)')
    Ω = mortar(((-1).^(1:N) .* Fill([1 0; 0 1], N))')

    vcat(map(vec,[Σ*X*E - I, Σ*Y*E, Ω*X*E, Ω*Y*E - [0 -1; -1 0],  X*Y - Y*X, X^2 + Y^2 - I])...)
end

@testset "Newton" begin
    @testset "circle" begin
        p = randn(14)
        for _ = 1:10
            J = ForwardDiff.jacobian(F_circle,p); p = p - (J \ F_circle(p))
        end
        Ax,Bx,Ay,By = unroll(p)
        @test norm(Ax) ≤ 10eps()
        @test Bx ≈ [0.5 0; 0 0.5]
        @test norm(Ay) ≤ 10eps()
        @test By ≈ [0 -0.5; 0.5 0]
    end
    @testset "semicircle" begin
        x0 = randn(14)
        for _ = 1:10
            J = ForwardDiff.jacobian(F_semicircle,x0); x0 = x0 - (J \ F_semicircle(x0))
        end
        Ax,Bx,Ay,By = unroll(x0)
        @test Ax ≈ Matrix(0.5I,2,2)
        @test Bx ≈ Matrix(0.25I,2,2)
        a₁₂ = (1 + sqrt(2))/4
        a₂₁ = (1 - sqrt(2))/4
        @test Ay ≈ [0 -0.5; -0.5 0]
        @test By ≈ [0 a₁₂; a₂₁ 0]
    end
end

@testset "Jacobi" begin
    x₂ = -(1/4)*sqrt(4+2*sqrt(2))
    x₀ = -1/sqrt(2)
    N = 100;
    X = BandedMatrix{Float64}(undef, (N,N), (2,2))
    X[band(0)] = [x₀ 1/2*ones(1,N-2) x₀]
    X[band(1)] .= X[band(-1)] .= 0;
    X[band(2)] = X[band(-2)] = [x₂ 1/4*ones(1,N-4) x₂]

    a₁₂ = (1 + sqrt(2))/4
    a₂₁ = (1 - sqrt(2))/4
    y₁=(1/4)*sqrt(4-2*sqrt(2))
    Y = BandedMatrix{Float64}(undef, (N,N), (3,3))
    Y[band(0)].=0
    Y[band(2)].=Y[band(-2)].=0
    d1 = [y₁ repeat([-1/2 a₂₁],1,Int64(round(N/2))-1)]
    d1[N-1] = y₁
    d3 = repeat([0 a₁₂],1,Int64(round(N/2)))
    Y[band(1)] = Y[band(-1)] = d1[1:N-1]
    Y[band(3)] = Y[band(-3)] = d3[1:N-3]

    @test X*Y ≈ Y*X
    @test X^2 + Y^2 ≈ Eye(N)
end


# function FF2(x)
#     Ax,Bx,Ay,By = unroll2(x)
#     N = 5
#     X = blocksymtricirculant(Ax, Bx, N)
#     Y = blocksymtricirculant(Ay, By, N)
#     E = mortar([zeros(2,2), [1 0; 0 1], Fill(zeros(2,2),N-2)...]')'
#     Σ = mortar(Fill([1 0; 0 1], N)')
#     Ω = mortar(((-1).^(1:N) .* Fill([1 0; 0 1], N))')

#     vcat(map(vec,[X*Y - Y*X, X^2 + Y^2 - I])...)
# end

# x0 = randn(14)
# J = ForwardDiff.jacobian(FF2,x0); x0 = x0 - (J \ FF2(x0))
# Ax,Bx,Ay,By = unroll2(x0)

# x = z -> Ax + (Bx/z + Bx'*z)
# y = z -> Ay + (By/z + By'*z)

# NN = 20; Z = Matrix{ComplexF64}(undef,2,NN)
# for (j,θ) in enumerate(range(0,2π; length=NN))
#     z = exp(θ*im)
#     _,Q = eigen(x(z))
#     Z[:,j] = (diag(real(Q'x(z)*Q)) + im*diag(real(Q'y(z)*Q)))
# end

# scatter(Z)


function F_circle(x)
    Bx = reshape(x,2,4)[:,1:2]
    By = reshape(x,2,4)[:,3:end]
    N = 5
    Z = zeros(eltype(x),2,2)
    X = blocksymtricirculant(Z, Bx, N)
    Y = blocksymtricirculant(Z, By, N)

    vcat(map(vec,[X*Y - Y*X, X^2 + Y^2 - I])...)
end

p = randn(8)
J = ForwardDiff.jacobian(F_circle,p); p = p - (J \ F_circle(p)); norm(F_circle(p))

Bx = reshape(p,2,4)[:,1:2]
By = reshape(p,2,4)[:,3:end]




x = z -> (Bx/z + Bx'*z)
y = z -> (By/z + By'*z)
NN = 20; Z = Matrix{ComplexF64}(undef,size(Bx,1),NN)
for (j,θ) in enumerate(range(0,2π; length=NN))
    z = exp(θ*im)
    λ,Q = eigen(x(z))
    Z[:,j] = λ  .+ im*real(diag(Q'*y(z)*Q))
end

@test real(Z).^2 + imag(Z).^2 ≈ Ones(2,NN)

scatter(vec(Z))

using BlockArrays
import BlockBandedMatrices: _BlockBandedMatrix

_BlockBandedMatrix(Vcat(0,mortar(Fill([1,0,0,1],∞))), [1; Fill(2,∞)], [1; Fill(2,∞)], (0,0))