using RandomMatrices
using RandomMatrixDistributions

using Plots
gr()

using LinearAlgebra

TRAINING_SAMPLES = 1#100000
TEST_SAMPLES = 1#10000
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
    return tridrand(Wigner(1), 3)
end

function plotDist()
    values = []
    for i in 1:100
        append!(values, sampleEigenvalues())
    end
    savefig(histogram(values, bins=100, title="Dist Eigenvalues"), "dist.png")
end

function plotMatrix()
    values = []
    for i in 1:100
        append!(values, eigvals(sampleMatrix()))
    end
    savefig(histogram(values, bins=100, title="Matrix Eigenvalues"), "matrix.png")
end

function generateTrainingData()
    training = []
    for i in 1:TRAINING_SAMPLES
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
        A = sampleMatrix()

        σs = svdvals(A)

        push!(testing, (A, σs))
    end

    return testing
end

plotDist()
plotMatrix()

exit()

trainingData = generateTrainingData()
testingData = generateTestingData()
println(trainingData, testingData)