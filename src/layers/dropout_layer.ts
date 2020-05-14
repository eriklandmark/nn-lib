import Layer from "./layer";
import Matrix from "../matrix";
import {SavedLayer} from "../model";

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

    feedForward(input: Layer | Matrix, isInTraining: boolean) {
        this.activation = (<Layer>input).activation
        if (isInTraining) {
            (<Matrix> this.activation).iterate((i: number, j: number) => {
                if(Math.random() < this.rate) {
                    (<Matrix> this.activation).set(i,j, 0)
                }
            })
        }
    }

    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix) {
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