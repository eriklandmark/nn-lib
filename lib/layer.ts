import Matrix from "./matrix";
import Vector from "./vector";

export default class Layer {
    weights: Matrix
    bias: Vector
    errorWeights: Matrix
    errorBias: Matrix
    output_error: Matrix
    activation: Matrix
    actFunc: Function
    actFuncDer: Function
    layerSize: number

    z: Matrix

    constructor(layerSize: number, prevLayerSize: number, actFunc: Function, actFuncDer: Function) {
        this.layerSize = layerSize
        this.weights = new Matrix()
        this.weights.createEmptyArray(prevLayerSize, layerSize)
        this.bias = new Vector(layerSize)
        this.errorWeights = new Matrix()
        this.errorBias = new Matrix()
        this.output_error = new Matrix()
        this.activation = new Matrix()
        this.actFunc = actFunc
        this.actFuncDer = actFuncDer
        this.z = new Matrix()
    }

    public populate() {
        this.weights.populateRandom();
        this.bias.populateRandom();
    }

    public feedForward(input: Layer | Matrix) {
        let act
        if (input instanceof Matrix) {
            if (this.activation.empty()) {
                this.activation.createEmptyArray(input.dim().r, this.layerSize)
            }
            act = input
        } else {
            if (this.activation.empty()) {
                this.activation.createEmptyArray(input.activation.dim().r, this.layerSize)
            }

            act = (<Layer> input).activation
        }
        this.z = <Matrix> act.mm(this.weights)

        this.z.iterate((i,j) => {
            this.z.set(i,j, this.z.get(i,j) + this.bias.get(j))
        })
        this.activation = this.actFunc(this.z)
    }

    updateWeights(l_rate: number) {
        this.weights = this.weights.sub(this.errorWeights.mul(l_rate))
        this.bias.iterate((val: number, i: number) => {
            this.bias.set(i, val - (this.errorBias.get(0, i) * l_rate))
        })
    }
}