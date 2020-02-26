import Matrix from "./matrix";
import Vector from "./vector";
import Activations from "./activations";

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
        this.z = act.mm(this.weights)

        /*this.z.iterate((i,j) => {
            this.z.set(i,j, this.z.get(i,j) + this.bias.get(i))
        })*/
        this.activation = this.actFunc(this.z)
        //console.log(this.weights.toString(10))
    }

    updateWeights(l_rate: number) {
        this.weights = this.weights.sub(this.errorWeights.mul(l_rate))
        /*const sums = <Matrix> this.errorBias.sum(1);
        this.bias.iterate((val: number, i: number) => {
            this.bias.set(i, val - (sums.get(i,0) * l_rate))
        })*/
    }
}