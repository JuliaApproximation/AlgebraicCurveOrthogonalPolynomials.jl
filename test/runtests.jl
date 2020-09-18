using OrthogonalPolynomialsAlgebraicCurves, LinearAlgebra, BlockBandedMatrices, BlockArrays, FillArrays, Test

include("test_semicircle.jl")

@testset "Quartic" begin
    @testset "Commutating properties" begin
        X,Y = quarticjacobi(10)
        @test (X*Y)[Block.(1:11), Block.(1:11)] ≈ (Y*X)[Block.(1:11), Block.(1:11)]
        @test (X^4 + Y^4)[Block.(1:10), Block.(1:10)] ≈ I(34)
    end

    @testset "Toeplitz SVD" begin
        X,Y = quarticjacobi(30)
        K = 25; σ1 = svdvals(Float64.(X[Block(K,K+1)]))
        K = 26; σ2 = svdvals(Float64.(X[Block(K,K+1)]))
        @test σ1 ≈ σ2 rtol=1E-2

        K = 25; σ1 = svdvals(Float64.(Y[Block(K,K+1)]))
        K = 26; σ2 = svdvals(Float64.(Y[Block(K,K+1)]))
        @test σ1 ≈ σ2 rtol=1E-2
    end
    
    @testset "32-by-32 symbols" begin
        periods = 6
        n=8*periods+7
        @time  X,Y= setprecision(300) do
            quarticjacobi(n)
        end
        Ax = BlockBandedMatrix(zeros(Float64,32,32), fill(4,8),fill(4,8), (1,1))
        for b = 0:6
            Ax[Block(b+1,b+2)]=X[Block(n-6+b,n-5+b)]
            Ax[Block(b+2,b+1)]=X[Block(n-5+b,n-6+b)]
        end
        Bx=BlockArray(zeros(Float64,32,32), fill(4,8), fill(4,8))
        Bx[Block(8,1)]=X[Block(n+1,n+2)]
        Ay = BlockBandedMatrix(zeros(Float64,32,32), fill(4,8),fill(4,8), (1,1))
        for b = 0:6
            Ay[Block(b+1,b+2)]=Y[Block(n-6+b,n-5+b)]
            Ay[Block(b+2,b+1)]=Y[Block(n-5+b,n-6+b)]
        end
        By=BlockArray{Float64}(zeros(Float64,32,32), fill(4,8), fill(4,8))
        By[Block(8,1)]=Y[Block(n+1,n+2)]
        θ=pi/2;
        z=exp(im*θ)
        xz = Bx'/z+Ax+Bx*z
        yz = By'/z+Ay+By*z
        @test xz*yz ≈ yz*xz rtol = 1E-2
        @test xz^4+yz^4≈I rtol=1E-2

        @test norm(Ax*Ay-Ay*Ax + Bx*By'-By'*Bx + Bx'*By - By*Bx') ≤ 0.005 
    end
end

# z = exp(0.1im)

# Xz = (Ax + Bx/z + z*Bx')
# Yz = (Ay + By/z + z*By')

# _,Q = Ay + By/z + z*By' .|> ComplexF64 |> Matrix |> Hermitian |> eigen;
# scatter(Float64.(real.(diag(Q'*(Ax + Bx/z + Bx'*z)*Q))), Float64.(real.(diag(Q'*(Ay + By/z + By'*z)*Q))))




# zeros(eltype(x),4,4)

# function _quartic_unroll_x(p)
#     Ax = BlockBandedMatrix(zeros(eltype(p),32,32), fill(4,8),fill(4,8), (1,1))
#     P = reshape(p,6,8)
#     for K = 1:7
#         view(view(Ax,Block(K,K+1)),band(0)) .= P[1:4,K]
#         view(view(Ax,Block(K,K+1)),band(-2)) .= P[5:end,K]
#         Ax[Block(K+1,K)] .= Ax[Block(K,K+1)]'
#     end
#     Bx = BlockArray(zeros(eltype(p),32,32), fill(4,8), fill(4,8))    
#     view(view(Bx,Block(8,1)),band(0)) .= P[1:4,end]
#     view(view(Bx,Block(8,1)),band(-2)) .= P[5:end,end]
#     Ax,Bx
# end

# function _quartic_unroll_y(p)
#     Ay = BlockBandedMatrix(zeros(eltype(p),32,32), fill(4,8),fill(4,8), (1,1))
#     P = reshape(p,7,8)
#     for K = 1:7
#         view(view(Ay,Block(K,K+1)),band(1)) .= P[1:3,K]
#         view(view(Ay,Block(K,K+1)),band(-1)) .= P[4:6,K]
#         view(view(Ay,Block(K,K+1)),band(-3)) .= P[7,K]
#         Ay[Block(K+1,K)] .= Ay[Block(K,K+1)]'
#     end
#     By = BlockArray(zeros(eltype(p),32,32), fill(4,8), fill(4,8))    
#     view(view(By,Block(8,1)),band(1)) .= P[1:3,end]
#     view(view(By,Block(8,1)),band(-1)) .= P[4:6,end]
#     view(view(By,Block(8,1)),band(-3)) .= P[7,end]
#     Ay,By
# end

# quartic_unroll(p) = tuple(_quartic_unroll_x(p[1:48])..., _quartic_unroll_y(p[49:end])...)


# function _quartic_roll_x(Ax,Bx)
#     ret = Vector{eltype(Ax)}()
#     for K = 1:7
#         append!(ret, Ax[Block(K,K+1)][band(0)])
#         append!(ret, Ax[Block(K,K+1)][band(-2)])
#     end
#     append!(ret, Bx[Block(8,1)][band(0)])
#     append!(ret, Bx[Block(8,1)][band(-2)])
#     ret
# end

# function _quartic_roll_y(Ay,By)
#     ret = Vector{eltype(Ay)}()
#     for K = 1:7
#         append!(ret, Ay[Block(K,K+1)][band(1)])
#         append!(ret, Ay[Block(K,K+1)][band(-1)])
#         append!(ret, Ay[Block(K,K+1)][band(-3)])
#     end
#     append!(ret, By[Block(8,1)][band(1)])
#     append!(ret, By[Block(8,1)][band(-1)])
#     append!(ret, By[Block(8,1)][band(-3)])
#     ret
# end

# quartic_roll(Ax,Bx,Ay,By) = [_quartic_roll_x(Ax,Bx); _quartic_roll_y(Ay,By)]

# @test _quartic_unroll_x(_quartic_roll_x(Ax,Bx)) == (Ax,Bx)
# @test all(_quartic_unroll_y(_quartic_roll_y(Ay,By)) .≈ (Ay,By))

# p0 = quartic_roll(Ax,Bx,Ay,By)
# @test all(quartic_unroll(p) .≈ (Ax,Bx,Ay,By))
# p0 = p
# function F_quartic(p)
#     Ax,Bx,Ay,By = quartic_unroll(p)
#     N = 9
#     n = size(Ax,1)
#     X = blocksymtricirculant(Ax, Bx, N)
#     Y = blocksymtricirculant(Ay, By, N)
#     # II = Matrix(I,n,n)
#     # ZZ = zeros(n,n)
#     # E = mortar([ZZ, II, Fill(ZZ,N-2)...]')'
#     # Σ = mortar(Fill(II, N)') # evaluate at 1
#     # Ω = mortar(((-1).^(1:N) .* Fill(II, N))') # evaluate at -1

#     # vcat(map(vec,[Σ*X*E - I, Σ*Y*E, Ω*X*E + I, Ω*Y*E ,  X*Y - Y*X, X^4 + Y^4 - I])...)
#     vcat(map(vec,[X*Y - Y*X, X^4 + Y^4 - I])...)
# end

# p = p0

# for _=1:10
#     global p
#     J = ForwardDiff.jacobian(F_quartic,p); p = p - (J \ F_quartic(p)); @show norm(F_quartic(p))
# end






# p = quartic_roll(Ax,Bx,Ay,By )

# p = randn(2* 6 * 8)
# Ax,Bx,Ay,By = quartic_unroll(p)

# Ax,Bx,Ay,By = quartic_unroll(p)

# x = z -> Ax + Bx/z + z*Bx'
# y = z -> Ay + By/z + z*By'
# z = exp(0.1im)
# x(z)y(z) - y(z)x(z) |> Matrix |> norm
# x(z)^4 + y(z)^4 - I |> Matrix |> norm

# Bx
# eigvals(x(1) |> Matrix)
# eigvals(y(1) |> Matrix)

# NN = 20; Z = Matrix{ComplexF64}(undef,size(Bx,1),NN)
# for (j,θ) in enumerate(range(0,2π; length=NN))
#     z = exp(θ*im)
#     λ,Q = eigen(Matrix(x(z)+im*y(z)))
#     Z[:,j] = λ
# end
# p

# using DeliminatedFiles
# writedlm("Ax.csv",Matrix(Ax))
# writedlm("Ay.csv",Matrix(Ay))
# writedlm("Bx.csv",Matrix(Bx))
# writedlm("By.csv",Matrix(By))

# @test real(Z).^4 + imag(Z).^4 ≈ Ones(32,NN)

# scatter(Z)