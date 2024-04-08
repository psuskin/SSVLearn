using JLD
using BSON
using Plots
using Statistics

using Flux
using NNlib
println("Loaded modules")

ONLY_LOWEST = true

version = "initial"
DIR = "experiments/$version"

data = load("data.jld")
testingData = data["testing"]

x = [x[1] for x in testingData]
y = [y[2] for y in testingData]

println("Loaded data")

model = Nothing
BSON.@load "$DIR/model512$(ONLY_LOWEST ? "s" : "").bson" model
if model == Nothing
    println("Model not found")
    exit()
end

println("Loaded model")

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
        y_pred = model(reshape(x[i].data, 9))
        savefig(plot([1, 2, 3], [y[i], y_pred], title="Singular values", label=["True" "Pred"]), "$DIR/plots/singular$i.png")
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
end

function testLoss(x, y, model)
    if ONLY_LOWEST
        return
    end

    losses = []
    for i in eachindex(x)
        push!(losses, Flux.mae(model(reshape(x[i].data, 9)), y[i], agg=sum))
    end
    println("Loss - ALL SINGULAR VALUES")
    println("Mean loss: ", mean(losses))
    println("Std loss: ", std(losses))
    println("Max loss: ", maximum(losses))
    println("Min loss: ", minimum(losses))
end

function testLossLowestSingular(x, y, model)
    losses = []
    for i in eachindex(x)
        if ONLY_LOWEST
            push!(losses, abs(model(reshape(x[i].data, 9))[1] - y[i][3]))
        else
            push!(losses, abs(model(reshape(x[i].data, 9))[3] - y[i][3]))
        end
    end
    println("Loss - LOWEST SINGULAR VALUE")
    println("Mean loss: ", mean(losses))
    println("Std loss: ", std(losses))
    println("Max loss: ", maximum(losses))
    println("Min loss: ", minimum(losses))
end


plotLoss()
# plotSingularValues(x, y, model)
testAnalysis(y)
testLoss(x, y, model)
testLossLowestSingular(x, y, model)