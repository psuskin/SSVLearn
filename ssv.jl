using Flux
using BSON

model = nothing
BSON.@load "model.bson" model

# Function to convert model parameters to Float64
function convert_to_float64(model)
    new_layers = []
    for layer in model
        if isa(layer, Dense)
            new_layer = Dense(
                convert(Array{Float64}, layer.weight),
                convert(Array{Float64}, layer.bias),
                layer.σ
            )
            push!(new_layers, new_layer)
        else
            push!(new_layers, layer)
        end
    end
    return Chain(new_layers...)
end

# Convert model parameters to Float64
modelSSV = convert_to_float64(model)

input = convert(Vector{Float64}, [0.0003610284291540382, -0.0006100995232036598, 0.00010790537134836162, -0.0007273238210451945, 0.00043271807731451933, 0.0007588973295521884])
# input = reshape([0.0003610284291540382 -0.0006100995232036598 0.00010790537134836162; -0.0006893540345596556 -0.0007273238210451945 0.00043271807731451933; 9.374998423902039e-5 0.0001558424160287082 0.0007588973295521884], 9)
output = modelSSV(input)
println(output)
println(typeof(output))

BSON.@save "modelSSV.bson" modelSSV