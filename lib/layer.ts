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
    dActivations: Matrix
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
        this.dActivations = new Matrix()
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
        //console.log(this.bias.toString())
        /*this.z.iterate((i,j) => {
            this.z.set(i,j, this.z.get(i,j) + this.bias.get(i))
        })*/
        this.activation = this.actFunc(this.z)
    }

    updateWeights(l_rate: number) {
        const newWeights = this.weights.copy()
        newWeights.iterate((i, j) => {
            newWeights.set(i, j, newWeights.get(i,j) - (l_rate * newWeights.get(i,j)));
        })
        this.weights = newWeights
        /*const sums = <Matrix> this.errorBias.sum(1);
        this.bias.iterate((val: number, i: number) => {
            this.bias.set(i, val - (sums.get(i,0) * l_rate))
        })*/
    }
}