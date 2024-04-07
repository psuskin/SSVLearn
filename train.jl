using Flux
using JLD
using BSON
println("Loaded modules")

ONLY_LOWEST = true

data = load("data.jld")
trainingData = data["training"]
# println(trainingData[1])

x = [reshape(x[1], 9) for x in trainingData]
y = [ONLY_LOWEST ? y[2][3] : y[2] for y in trainingData]
# println(x[1], y[1])

split = 0.8
split_index = round(Int, split*length(x))
x_train = x[1:split_index]
y_train = y[1:split_index]
x_validation = x[split_index+1:end]
y_validation = y[split_index+1:end]

println(size(x_train), size(y_train), size(x_validation), size(y_validation))
println(size(x_train[1]), size(y_train[1]))

println("Loaded data")

# Anything that is commented out below is part of an implementation which yielded lower accuracy than the current implementation.

# MODEL ARCHITECTURE:
model_size = "large"
# model_size = "small"

if model_size == "small"
    model = Chain(
        Dense(9, 8, relu),
        Dense(8, 7, relu),
        Dense(7, 6, relu),
        Dense(6, 5, relu),
        Dense(5, 3)
    )
elseif model_size == "large"
    model = Chain(
        Dense(9, 100, relu),
        Dense(100, 75, relu),
        Dense(75, 25, relu),
        Dense(25, 10, relu),
        Dense(10, ONLY_LOWEST ? 1 : 3)
    )
end

# LOSS FUNCTION:
loss(m, x, y) = Flux.mae(m(x), y, agg=sum)
# loss(m, x, y) = Flux.mse(m(x), y)

# OPTIMIZER: 
opt = Flux.setup(Descent(0.001), model)
# opt = Flux.setup(Adam(), model)
# opt = Flux.setup(Descent(), model)

function compound_loss(m, x, y)
    losses = []
    for i in eachindex(x)
        push!(losses, loss(m, x[i], y[i]))
    end
    return sum(losses) / length(losses)
end

println("Training model")

# n_epochs = 256
n_epochs = 512
println("Epoch: 0, trainingloss: ", compound_loss(model, x_train, y_train), " | validation loss: ", compound_loss(model, x_validation, y_validation))
for epoch in 1:n_epochs
    Flux.train!(loss, model, zip(x_train, y_train), opt)
    println("Epoch: $epoch, trainingloss: ", compound_loss(model, x_train, y_train), " | validation loss: ", compound_loss(model, x_validation, y_validation))
end

println("Saving model")

BSON.@save "model.bson" model