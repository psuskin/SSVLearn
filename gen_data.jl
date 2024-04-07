using RandomMatrices
using RandomMatrixDistributions

using Plots
gr()

using LinearAlgebra

using JLD

TRAINING_SAMPLES = 100000
TEST_SAMPLES = 10000
N = 3

function sampleEigenvalues()
    if false
        λs = randeigvals(SpikedWishart(2, 10, N))
        for j in 1:N
            λs[j] = rand(Bool) ? λs[j] : -λs[j]
        end
    else
        λs = eigvalrand(Wigner(1), 3)
    end

    return λs#sort(λs)
end

function sampleMatrix()
    return rand(Wigner(1), 3)
end

function plotEigenvalues()
    dist = []
    matrix = []
    for i in 1:10000
        append!(dist, sampleEigenvalues())
        append!(matrix, eigvals(sampleMatrix()))
    end
    savefig(histogram([dist, matrix], bins=100, title="Eigenvalues", label=["dist" "matrix"], fillcolor=[:red :black], fillalpha=0.2, normalize=:pdf), "eigenvalues.png")
end

function plotMatrixValues()
    dist = []
    matrix = []
    for i in 1:10000
        Q = rand(Haar(1), 3)
        append!(dist, reshape(Q * Diagonal(sampleEigenvalues()) * Q', 9))

        append!(matrix, reshape(sampleMatrix(), 9))
    end
    savefig(histogram([dist, matrix], bins=100, title="Matrix values", label=["dist" "matrix"], fillcolor=[:red :black], fillalpha=0.2, normalize=:pdf), "matrixValues.png")
end

function generateTrainingData()
    training = []
    for i in 1:TRAINING_SAMPLES
        if i % 1000 == 0
            println(i)
        end

        # Initialize eigenvalues using passed distribution
        λs = sampleEigenvalues()

        # Initialize eigenvectors using Haar distribution
        Q = rand(Haar(1), 3)

        A = Q * Diagonal(λs) * Q'
        σs = sort(abs.(λs), rev=true)

        push!(training, (A, σs))
    end

    return training
end

function generateTestingData()
    testing = []
    for i in 1:TEST_SAMPLES
        if i % 1000 == 0
            println(i)
        end

        # Initialize matrix using Gaussian orthogonal ensemble (https://github.com/JuliaMath/RandomMatrices.jl/blob/6b01eb2cb3c6cb2cf2d5676e8d1223022f3a8d9c/src/GaussianEnsembles.jl#L48)
        A = sampleMatrix()

        σs = svdvals(A)

        push!(testing, (A, σs))
    end

    return testing
end

function analyze()
    if true
        plotEigenvalues()
        plotMatrixValues()
    end
end

if true
    analyze()
else
    trainingData = generateTrainingData()
    testingData = generateTestingData()
    # println(trainingData, testingData)
    save("data.jld", "training", trainingData, "testing", testingData)
end