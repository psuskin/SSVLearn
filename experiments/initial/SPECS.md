## Model size
Chain(
    Dense(9, 100, relu),
    Dense(100, 75, relu),
    Dense(75, 25, relu),
    Dense(25, 10, relu),
    Dense(10, ONLY_LOWEST ? 1 : 3)
)

## Loss function
Flux.mae(m(x), y, agg=sum)

## Optimization algorithm
Descent(0.001)