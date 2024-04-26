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
println()

# Build basis for the orthogonal space
function gram_schmidt_orthogonal_basis(v)
    v_normalized = v / norm(v)
    
    basis = [v_normalized]
    
    # Gram-Schmidt process
    for i in 2:length(v)
        u = rand(length(v))
        
        # Orthogonalize against all previous basis vectors
        for j in 1:i-1
            u -= dot(u, basis[j]) / dot(basis[j], basis[j]) * basis[j]
        end
        
        # Normalize the resulting orthogonal vector
        u /= norm(u)

        push!(basis, u)
    end
    
    return basis[2:end]
end
orthogonal_basis = gram_schmidt_orthogonal_basis(grad)
println("Example basis for the subspace orthogonal to the gradient:")
display(orthogonal_basis)
println()

# Ensure that the basis is orthogonal (including the gradient)
println("all(dot(basis[i], basis[j]) < 1e5 for i in 1:length(basis), j in 1:length(basis)) end = ", let basis = vcat([grad], orthogonal_basis); all(dot(basis[i], basis[j]) < 1e5 for i in 1:length(basis), j in 1:length(basis)) end, ", where basis = vcat([grad], orthogonal_basis))")
println()

# Compute the output for the basis vectors
for i in 1:length(orthogonal_basis)
    println("Output for input shifted in direction of basis vector $i (ϵ=0.01): ",  model(input + 0.01f0 * orthogonal_basis[i]))
end