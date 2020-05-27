using DelimitedFiles, BlockArrays, Test

Ax = PseudoBlockArray(readdlm("experiments/Ax.csv"), fill(4,8), fill(4,8))
Bx = PseudoBlockArray(readdlm("experiments/Bx.csv"), fill(4,8), fill(4,8))
Ay = PseudoBlockArray(readdlm("experiments/Ay.csv"), fill(4,8), fill(4,8))
By = PseudoBlockArray(readdlm("experiments/By.csv"), fill(4,8), fill(4,8))


X = z -> Ax + Bx/z + z*Bx'
Y = z -> Ay + By/z + z*By'

 z = exp(0.1im); norm(X(z)Y(z) - Y(z)X(z))
 z = exp(0.1im); norm(X(z)^4 + Y(z)^4 - I)

Ax[Block(1,2)] |> svdvals
Ax[Block(2,3)] |> svdvals
Ax[Block(3,4)] |> svdvals
Ax[Block(4,5)] |> svdvals
Ax[Block(5,6)] |> svdvals
Ax[Block(6,7)] |> svdvals
Ax[Block(7,8)] |> svdvals
Bx[Block(8,1)] |> svdvals

norm(Ax*Ay-Ay*Ax + Bx*By' + Bx'*By - By*Bx' - By'*Bx)
norm(Bx*By-By*Bx)

@test Bx^4 ≈ -By^4
@test X(z)^2 ≈ Bx^2/z^2 + (Ax*Bx + Bx*Ax)/z + Ax^2 + Bx*Bx' + Bx'*Bx + z*(Bx'*Ax + Ax*Bx') + z^2*(Bx^2)'

