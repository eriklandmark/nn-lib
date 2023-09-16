import Layer from "./layer";
import { IActivation } from "../activations/activations";
import Tensor from "../tensor";
import { SavedLayer } from "../model";
export default class DenseLayer extends Layer {
    layerSize: number;
    ff_kernel: any;
    act_kernel: any;
    bp_error_kernel: any;
    bp_error_weight_kernel: any;
    weights: Tensor;
    errorWeights: Tensor;
    bias: Tensor;
    constructor(layerSize?: number, activation?: IActivation);
    buildLayer(prevLayerShape: number[]): void;
    feedForward(input: Layer | Tensor, isInTraining: boolean): void;
    backPropagation(prev_layer: Layer, next_layer: Layer | Tensor): void;
    toSavedModel(): SavedLayer;
    fromSavedModel(data: SavedLayer): void;
}
