import Layer from "./layer";
import Matrix from "../../matrix";
import DenseLayer from "./dense_layer";
import Sigmoid from "../activations/sigmoid";

export default class DropoutLayer extends DenseLayer {

    rate: number = 0

    constructor(rate: number) {
        super(0, new Sigmoid());
        this.rate = rate
    }

    buildLayer(prevLayerShape: number[]) {
        this.layerSize = prevLayerShape[0];
        super.buildLayer(prevLayerShape);
    }

    feedForward(input: Layer | Matrix, isInTraining: boolean) {
        this.activation = (<Layer>input).activation
        if (isInTraining) {
            this.activation.iterate((i: number, j: number) => {
                if(Math.random() < this.rate) {
                    this.activation.set(i,j, 0)
                }
            })
        }
    }

    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix) {
        this.weights = prev_layer.weights;
        this.output_error = prev_layer.output_error;
    }

    updateWeights(l_rate: number) {}
}