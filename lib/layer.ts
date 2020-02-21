import Matrix from "./matrix";
import Vector from "./vector";
import Activations from "./activations";

export default class Layer {
    weights: Matrix
    bias: Vector
    error: Vector
    output_error: Vector
    activation: Vector
    dActivations: Vector

    constructor(layerSize: number, prevLayerSize: number) {
        this.weights = new Matrix()
        this.weights.createEmptyArray(layerSize, prevLayerSize)
        this.bias = new Vector(layerSize)
        this.error = new Vector(layerSize)
        this.output_error = new Vector(layerSize)
        this.activation = new Vector(layerSize)
        this.dActivations = new Vector(layerSize)
    }

    public populate() {
        this.weights.populateRandom();
        this.bias.populateRandom();
    }

    public feedForward(input: Layer | Vector) {
        let act: Vector
        if (input instanceof Vector) {
            act = input
        } else {
            act = (<Layer> input).activation
        }
        this.activation = <Vector> Activations.sigmoid( (<Vector> this.weights.mm(act)).add(this.bias))
    }

    updateWeights(l_rate) {
        const newWeights = this.weights.copy()
        newWeights.iterate((i, j) => {
            newWeights.set(i, j, newWeights.get(i,j) - (l_rate * this.error.get(i)));
        })
        this.weights = newWeights
    }
}