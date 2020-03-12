import Layer from "./layer";
import Tensor from "../../tensor";
import IActivation from "../activations/activations";
import Matrix from "../../matrix";

export default class FlattenLayer extends Layer {

    constructor() {
        super();
    }

    buildLayer(prevLayerShape: number[]) {

    }

    feedForward(input: Layer, isInTraining: boolean) {

    }

    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix) {

    }
}