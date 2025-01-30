using Flux
using Plots
using ChainPlots

model = Chain(
    Dense(6, 100, relu),
    Dense(100, 75, relu),
    Dense(75, 25, relu),
    Dense(25, 10, relu),
    Dense(10, 1)
)

ChainPlots.plot(model)
savefig("model.pdf")