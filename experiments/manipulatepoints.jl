using Pkg, ClassicalOrthogonalPolynomials, OrthogonalPolynomialsAlgebraicCurves, Plots, LinearAlgebra, Interact, InfiniteLinearAlgebra

# Semicircle


Aˣ = Matrix(0.5I,2,2)
Bˣ = Matrix(0.25I,2,2)
a₁₂ = (1 + sqrt(2))/4
a₂₁ = (1 - sqrt(2))/4
Aʸ = [0 -0.5; -0.5 0]
Bʸ = [0 a₁₂; a₂₁ 0]
X = HermLaurent(Aˣ,Bˣ)
Y = HermLaurent(Aʸ,Bʸ)

T = BlockTridiagonal(X)

U,L = ul(T-1.1I, Val(false))
N = 100; U[Block.(1:N),Block.(1:N)]*L[Block.(1:N),Block.(1:N)]
L[Block.(1:N),Block.(1:N)] |> inv
MemoryLayout(L)

C,A,B = (T-2I)[Block(2,1)],(T-2I)[Block(2,2)],(T-2I)[Block(2,3)]

Z = vec(specgrid(X, Y)); Z = Z[sortperm(imag(Z))]

@manipulate for θ in range(0,π;length=100)
	z = exp(im*θ)
	p = plot(real.(Z), imag.(Z))
	scatter!(jointeigvals(X[z],Y[z]))
end


# Square
Bˣ = [-0.5 0 0 0; 0 0 0 1; 0 0 -0.5 0; 0 0 0 0]
Bʸ = [0 0 -1 0; 0 0.5 0 0; 0 0 0 0; 0 0 0 0.5]
X = HermLaurent(zero.(Bˣ), Bˣ)
Y = HermLaurent(zero.(Bʸ), Bʸ)

T = BlockTridiagonal(X)
C,A,B = (T-2I)[Block(2,1)],(T-2I)[Block(2,2)],(T-2I)[Block(2,3)]
ul(T-2I,Val(false))
A


histogram(eigvals(blocksymtricirculant(X, 1000) |> Matrix); nbins=100)

@manipulate for θ in range(0,π;length=100)
	z = exp(im*θ)
	p = plot([-1,1,1,-1,-1],[-1,-1,1,1,-1])
	scatter!(jointeigvals(X[z][[1,3],[1,3]],Y[z][[1,3],[1,3]]))
	scatter!(jointeigvals(X[z][[2,4],[2,4]],Y[z][[2,4],[2,4]]))
end





z = exp(0.1*im)
X[z]






Y[z]
