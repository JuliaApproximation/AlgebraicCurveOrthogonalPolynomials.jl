using InfiniteLinearAlgebra, BlockBandedMatrices, BandedMatrices, BlockArrays, InfiniteArrays, FillArrays, LazyArrays, Test


import BandedMatrices: _BandedMatrix
B = brand(5,5,1,1)
d = Vcat([zeros(1,1)], Fill(zeros(2,2), ∞))
import BlockArrays: _BlockArray
_BlockArray(_BandedMatrix(permutedims(Hcat(Vcat([fill(NaN,1,1),fill(1.0,1,2)], Fill(Matrix(1.0I,2,2), ∞)),
                    d,
                    Vcat([fill(1.0,2,1),Matrix(1.0I,2,2)], Fill(Matrix(1.0I,2,2), ∞)))), ℵ₀, 1, 1), size.(d,1), size.(d,2))

BlockBandedMatrix(A)
isb

using ClassicalOrthogonalPolynomials

P = Legendre()
X = jacobimatrix(P)
c,a,b = X.dl,X.d,X.du

using LazyBandedMatrices
mortar(LazyBandedMatrices.Tridiagonal(Vcat([[1.0 0]], Matrix.(Diagonal.(vcat.(b,b)))),
                 Vcat([zeros(1,1)], Fill(zeros(2,2), ∞)),
                 Vcat([permutedims([1.0 0])], Matrix.(Diagonal.(vcat.(c,c))))))

d = Vcat([zeros(1,1)], Fill(zeros(2,2), ∞))
_BlockArray(_BandedMatrix(permutedims(Hcat(Vcat([fill(NaN,1,1), [1.0 0]], Matrix.(Diagonal.(vcat.(b,b)))),
                 d,
                 Vcat([permutedims([1.0 0])], Matrix.(Diagonal.(vcat.(c,c)))))), ℵ₀, 1, 1), size.(d,1), size.(d,2))


_BlockArray(_BandedMatrix(permutedims(Hcat(Vcat([fill(NaN,1,1), [1.0 0]], Matrix.(Diagonal.(vcat.(b,b)))),
                 d,
                 Vcat([zeros(2,1)], Fill(zeros(2,2), ∞)),
                 Vcat([permutedims([1.0 0])], Matrix.(Diagonal.(vcat.(c,c)))))), ℵ₀, 2, 1), size.(d,1), size.(d,2))



XB = unitblocks(X)
Z = Zeros(axes(XB))
BlockBroadcastArray{Float64}(Diagonal, XB, XB)
BlockBroadcastArray{Float64}(hvcat, 2, XB, Z, Z, XB)

