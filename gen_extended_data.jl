using Random
using LinearAlgebra
using RandomMatrices

using Plots
gr()

using JLD

TEST_SAMPLES = 10000

function uniformLargeMatrixValues()
    data = []
    for i in 1:TEST_SAMPLES
        randomMatrix = rand(Float32, (3, 3)) * 200 .- 100
        singulars = svdvals(randomMatrix)
        push!(data, (randomMatrix, singulars))
    end

    return data
end

function uniformEigenvalues()
    data = []
    for i in 1:TEST_SAMPLES
        randomEigenvalues = rand(Float32, 3) * 3 .- 1.5
        Q = rand(Haar(1), 3)
        A = Q * Diagonal(randomEigenvalues) * Q'
        σs = sort(abs.(randomEigenvalues), rev=true)
        push!(data, (A, σs))
    end

    return data
end

extended_datasets = Dict("uniform_large_matrix_values" => uniformLargeMatrixValues(), "uniform_eigenvalues" => uniformEigenvalues())

function plotEigenvalues()
    singularvalues1 = []
    singularvalues2 = []
    for i in 1:10000
        randomMatrix = rand(Float32, (3, 3)) * 2 .- 1
        append!(singularvalues1, svdvals(randomMatrix))

        randomEigenvalues = rand(Float32, 3) * 3 .- 1.5
        Q = rand(Haar(1), 3)
        A = Q * Diagonal(randomEigenvalues) * Q'
        append!(singularvalues2, svdvals(A))
    end

    savefig(histogram([singularvalues1, singularvalues2], bins=100, title="Singular values", label=["uniformLargeMatrixValues" "uniformEigenvalues"], fillcolor=[:red :black], fillalpha=0.2, normalize=:pdf), "singularvalues.png")
end

if true
    plotEigenvalues()
else
    extended_data = Dict()
    if "extended_data.jld" in readdir()
        extended_data = load("extended_data.jld")
    end

    for (key, value) in extended_datasets
        if key in keys(extended_data)
            # println("Key $key already exists in extended_data.jld")
            continue
        end

        extended_data[key] = value
    end

    save("extended_data.jld", extended_data)
end