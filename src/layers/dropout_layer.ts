import Layer from "./layer";
import {SavedLayer} from "../model";
import Tensor from "../tensor";

export default class DropoutLayer extends Layer {

    rate: number = 0
    type: string = "dropout"

    constructor(rate: number = 0.2) {
        super()
        this.rate = rate
    }

    buildLayer(prevLayerShape: number[]) {
        this.shape = prevLayerShape
    }

    feedForward(input: Layer, isInTraining: boolean) {
        this.activation = (<Layer>input).activation
        if (isInTraining) {
            this.activation.iterate((pos) => {
                if(Math.random() < this.rate) {
                    (<Tensor> this.activation).set(pos, 0)
                }
            }, true)
        }
    }

    backPropagation(prev_layer: Layer, next_layer: Layer) {
        this.weights = prev_layer.weights;
        this.output_error = prev_layer.output_error;
    }

    updateLayer() {}

    toSavedModel(): SavedLayer {
        const data = super.toSavedModel()
        data.layer_specific = {
            rate: this.rate
        }

        return data
    }

    fromSavedModel(data: SavedLayer) {
        super.fromSavedModel(data)
        this.rate = data.layer_specific.rate
    }
}