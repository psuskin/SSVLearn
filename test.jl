using JLD
using BSON
using Plots
using Statistics
using Random
using LinearAlgebra

using Flux
using NNlib
println("Loaded modules")

ONLY_LOWEST = true

version = "second"
DIR = "experiments/$version"

datasets = ["testing", "uniform_large_matrix_values", "uniform_eigenvalues"]
DATASET = datasets[1]

if DATASET == "testing"
    println("Testing original dataset")
    data = load("data.jld")
    testingData = data["testing"]
    x = [x[1].data for x in testingData]
else
    println("Testing extended dataset: $DATASET")
    data = load("extended_data.jld")
    testingData = data[DATASET]
    x = [x[1] for x in testingData]
end
y = [y[2] for y in testingData]

println("Loaded data")

model = Nothing
BSON.@load "$DIR/model512$(ONLY_LOWEST ? "s" : "").bson" model
if model == Nothing
    println("Model not found")
    exit()
end

baseline(x) = [1.0, 0.5, 0.0] # x * [1.0, 0.5, 0.0]

println("Loaded model")

println()

function plotLoss()
    open("$DIR/loss.txt", "r") do f
        currentPlot = ""
        trainingLoss = []
        validationLoss = []
        for line in eachline(f)
            if !contains(line, "Epoch")
                if !isempty(trainingLoss)
                    plot(trainingLoss, label="Training loss", title="Loss", xlabel="Epoch", ylabel="Loss", yaxis=:log)
                    plot!(validationLoss, label="Validation loss")
                    savefig("$DIR/plots/loss_$currentPlot.png")
                    trainingLoss = []
                    validationLoss = []
                end
                currentPlot = line
            else
                # Lines has this format: Epoch: 0, trainingloss: 1.7562293881121427 | validation loss: 1.7520427667608514
                append!(trainingLoss, parse(Float64, split(split(line, "|")[1], ":")[3]))
                append!(validationLoss, parse(Float64, split(split(line, "|")[2], ":")[2]))
            end
        end
        if !isempty(trainingLoss)
            plot(trainingLoss, label="Training loss", title="Loss", xlabel="Epoch", ylabel="Loss", yaxis=:log)
            plot!(validationLoss, label="Validation loss")
            savefig("$DIR/plots/loss_$currentPlot.png")
        end
    end
end

function plotSingularValues(x, y, model)
    if ONLY_LOWEST
        return
    end

    for i in 1:10
        y_pred = model(reshape(x[i], 9))
        savefig(plot([1, 2, 3], [y[i], y_pred], title="Singular values", label=["True" "Pred"]), "$DIR/plots/singular$i.png")
    end
end

function testRandom(model)
    # Set random seed
    Random.seed!(1234)

    # Random matrix with values between -100 and 100
    randomMatrix = rand(Float32, (3, 3)) * 200 .- 100
    singulars = svdvals(randomMatrix)
    println("Random matrix: ", randomMatrix)
    println("Singular values: ", singulars)

    # Predictions
    y_pred = model(reshape(randomMatrix, 9))
    y_baseline = baseline(randomMatrix)
    
    if ONLY_LOWEST
        println("Error: ", abs(y_pred[1] - singulars[3]), " | Baseline: ", abs(y_baseline[3] - singulars[3]))
    else
        println("Error: ", Flux.mae(y_pred, singulars, agg=sum), " | Baseline: ", Flux.mae(y_baseline, singulars, agg=sum))
        savefig(plot([1, 2, 3], [singulars, y_pred, y_baseline], title="Random matrix", label=["True" "Pred" "Baseline"]), "$DIR/plots/random.png")
    end
end

function testAnalysis(y)
    allSingulars = vec(y)
    println("All singular values")
    println("Mean: ", mean(allSingulars))
    println("Std: ", std(allSingulars))

    lowestSingulars = [s[3] for s in y]
    println("Lowest singular values")
    println("Mean: ", mean(lowestSingulars))
    println("Std: ", std(lowestSingulars))
    println("Max: ", maximum(lowestSingulars))
    println("Min: ", minimum(lowestSingulars))
    println()
end

function testLoss(x, y, model)
    if ONLY_LOWEST
        return
    end

    losses = []
    lossesBaseline = []
    for i in eachindex(x)
        push!(losses, Flux.mae(model(reshape(x[i], 9)), y[i], agg=sum))
        push!(lossesBaseline, Flux.mae(baseline(x[i]), y[i], agg=sum))
    end
    println("Loss - ALL SINGULAR VALUES")
    println("Mean loss: ", mean(losses), " | Baseline: ", mean(lossesBaseline))
    println("Std loss: ", std(losses), " | Baseline: ", std(lossesBaseline))
    println("Max loss: ", maximum(losses), " | Baseline: ", maximum(lossesBaseline))
    println("Min loss: ", minimum(losses), " | Baseline: ", minimum(lossesBaseline))
    println()
end

function testLossLowestSingular(x, y, model)
    losses = []
    lossesBaseline = []
    for i in eachindex(x)
        if ONLY_LOWEST
            push!(losses, abs(model(reshape(x[i], 9))[1] - y[i][3]))
        else
            push!(losses, abs(model(reshape(x[i], 9))[3] - y[i][3]))
        end
        push!(lossesBaseline, abs(baseline(x[i])[3] - y[i][3]))
    end
    println("Loss - LOWEST SINGULAR VALUE")
    println("Mean loss: ", mean(losses), " | Baseline: ", mean(lossesBaseline))
    println("Std loss: ", std(losses), " | Baseline: ", std(lossesBaseline))
    println("Max loss: ", maximum(losses), " | Baseline: ", maximum(lossesBaseline))
    println("Min loss: ", minimum(losses), " | Baseline: ", minimum(lossesBaseline))
    println()
end


# plotLoss()
# plotSingularValues(x, y, model)
# testRandom(model)
testAnalysis(y)
testLoss(x, y, model)
testLossLowestSingular(x, y, model)