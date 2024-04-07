using JLD
using BSON
using Plots
using Statistics

using Flux
using NNlib
println("Loaded modules")

data = load("data.jld")
testingData = data["testing"]

x = [x[1] for x in testingData]
y = [y[2] for y in testingData]

println("Loaded data")

model = Nothing
BSON.@load "model512.bson" model
if model == Nothing
    println("Model not found")
    exit()
end

println("Loaded model")

function plotSingularValues(x, y, model)
    for i in 1:10
        y_pred = model(reshape(x[i].data, 9))
        savefig(plot([1, 2, 3], [y[i], y_pred], title="Singular values", label=["True" "Pred"]), "singular$i.png")
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
        push!(losses, abs(model(reshape(x[i].data, 9))[3] - y[i][3]))
    end
    println("Loss - LOWEST SINGULAR VALUE")
    println("Mean loss: ", mean(losses))
    println("Std loss: ", std(losses))
    println("Max loss: ", maximum(losses))
    println("Min loss: ", minimum(losses))
end


# plotSingularValues(x, y, model)
testAnalysis(y)
testLoss(x, y, model)
testLossLowestSingular(x, y, model)