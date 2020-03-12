import Layer from "./layer";
import Tensor from "../../tensor";
import Matrix from "../../matrix";

export default class FlattenLayer extends Layer {

    constructor() {
        super();
    }

    prevShape: number[] = []

    buildLayer(prevLayerShape: number[]) {
        this.prevShape = prevLayerShape
        this.shape = [prevLayerShape.reduce((acc, val) => acc * val)];
    }

    feedForward(input: Layer, isInTraining: boolean) {
        const matrix = new Matrix((<Tensor[]>input.activation).map((tensor) => tensor.vectorize()))
        this.activation = matrix.transpose()
    }

    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix) {
        const dout = (<Matrix>prev_layer.output_error).mm(prev_layer.weights.transpose())
        console.log(dout.toString())
    }
}