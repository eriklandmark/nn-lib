import Layer from "./layer";
import Matrix from "../../matrix";

export default class ConvolutionLayer extends Layer {

    filterSize: number[] = []

    constructor(filterSize: number[]) {
        super();
        this.filterSize = filterSize
    }

    buildLayer(prevLayerShape: number[]) {
        super.buildLayer(prevLayerShape);
    }

    feedForward(input: Layer | Matrix, isInTraining: boolean) {
        this.activation = (<Layer>input).activation
    }

    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix) {
        this.weights = prev_layer.weights;
        this.output_error = prev_layer.output_error;
    }

    updateWeights(l_rate: number) {}
}