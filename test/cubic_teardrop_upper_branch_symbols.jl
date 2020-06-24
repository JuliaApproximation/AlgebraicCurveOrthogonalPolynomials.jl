using LinearAlgebra, Test
v = sqrt(2)/64
Bx = [0 0 0;1/8 0 0;1/2 -1/8 0]
By = [v 0 0;-6*v v 0;v 6*v v]
Ax = [-1/4 -1/2 -1/8;-1/2 -1/4 1/2;-1/8 1/2 -1/4]
Ay = [12*v -v 6*v;-v 12*v v;6*v v 12*v]
x(z) = Bx'/z + Ax + Bx*z
y(z) = By'/z + Ay + By*z
z=exp(im*0)
@testset "cubic teardrop upper branch symbols" begin
@test x(z)*y(z) ≈ y(z)*x(z)
@test y(z)^2 ≈ 1/4*(I - x(z))^2*(I + x(z))
end
