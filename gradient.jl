using BSON
using RandomMatrices
using LinearAlgebra
using Random

using Flux
using Tracker
println("Loaded modules")

model = Nothing
BSON.@load "model.bson" model
if model == Nothing
    println("Model not found")
    exit()
end

println("Loaded model")
println()

Random.seed!(12345)
matrix = rand(Wigner(1), 3)
matrix = convert(Array{Float32}, matrix)

input = let utv = vec(triu(matrix)); utv[utv .!= 0] end
println("Input: ", input)

output = model(input)
println("Predicted output: ", output)
println("Genuine output: ", svdvals(matrix)[3])
println()

# Compute gradient w.r.t. input
gradient = Tracker.withgradient(x -> model(x)[1], input)
grad = gradient[:grad][1]
grad = grad / norm(grad)
println("Gradient (normalized): ", grad)

# Shift input in the direction of the gradient
println("Output for input shifted in direction of gradient (ϵ=0.01): ",  model(input + 0.01f0 * grad))
println("Output for input shifted in opposite direction of gradient (ϵ=0.01): ",  model(input - 0.01f0 * grad))
println()

# Shift input orthogonal to the gradient
orthogonal = [1, 1, 1, 1, 1, - sum(grad[1:5]) / grad[6]]
orthogonal = orthogonal / norm(orthogonal)
println("Example vector orthogonal to gradient (normalized): ", orthogonal)
println("Output for input shifted orthogonally to gradient (ϵ=0.01): ",  model(input - 0.01f0 * orthogonal))