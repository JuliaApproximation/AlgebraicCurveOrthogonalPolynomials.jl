Ax = Matrix(0.5I,2,2)
Bx = Matrix(0.25I,2,2)
a₁₂ = (1 + sqrt(2))/4
a₂₁ = (1 - sqrt(2))/4
Ay = [0 -0.5; -0.5 0]
By = [0 a₁₂; a₂₁ 0]

X = z -> Ax + (Bx/z + Bx'*z)
Y = z -> Ay + (By/z + By'*z)

θ = range(-π/2,π/2;length=1000)
p = plot(cos.(θ), sin.(θ); label=nothing)

z = exp(0.5im); scatter!(jointeigvals(x(z),y(z))...)
z = exp(1im); scatter!(jointeigvals(x(z),y(z))...)
z = exp(π*im); scatter!(jointeigvals(x(z),y(z))...)
z = exp(3π/2*im); scatter!(jointeigvals(x(z),y(z))...)