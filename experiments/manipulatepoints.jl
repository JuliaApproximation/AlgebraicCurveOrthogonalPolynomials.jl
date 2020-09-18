### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 067cfe2c-f9c6-11ea-3212-992c5aeb05ed
using Pkg, OrthogonalPolynomialsQuasi, OrthogonalPolynomialsAlgebraicCurves, Plots, LinearAlgebra

# ╔═╡ f949b586-f9c4-11ea-11db-a9e4b56d5465
md"
This explores manipulating points on a HermLaurent
"

# ╔═╡ b341745e-f9c5-11ea-3d7a-c5deafbe1e77
Pkg.activate("/Users/sheehanolver/Documents/Supervision/Fasondini")

# ╔═╡ 56ed1e32-f9c6-11ea-2f2e-17ab6e22df70
md"# Semicircle"

# ╔═╡ 34efbc4e-f9c7-11ea-2398-d1e3790bec07
begin
	Aˣ = Matrix(0.5I,2,2)
	Bˣ = Matrix(0.25I,2,2)
	a₁₂ = (1 + sqrt(2))/4
	a₂₁ = (1 - sqrt(2))/4
	Aʸ = [0 -0.5; -0.5 0]
	Bʸ = [0 a₁₂; a₂₁ 0]
	X = HermLaurent(Aˣ,Bˣ)
	Y = HermLaurent(Aʸ,Bʸ)
end

# ╔═╡ Cell order:
# ╟─f949b586-f9c4-11ea-11db-a9e4b56d5465
# ╠═b341745e-f9c5-11ea-3d7a-c5deafbe1e77
# ╠═067cfe2c-f9c6-11ea-3212-992c5aeb05ed
# ╟─56ed1e32-f9c6-11ea-2f2e-17ab6e22df70
# ╠═34efbc4e-f9c7-11ea-2398-d1e3790bec07
