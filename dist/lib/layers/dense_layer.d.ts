import Layer from "./layer";
import Matrix from "../../matrix";
import { IActivation } from "../activations/activations";
import { SavedLayer } from "../../model";
export default class DenseLayer extends Layer {
    layerSize: number;
    ff_kernel: any;
    act_kernel: any;
    bp_error_kernel: any;
    bp_error_weight_kernel: any;
    constructor(layerSize?: number, activation?: IActivation);
    buildLayer(prevLayerShape: number[]): void;
    buildFFKernels(batch_size: number): void;
    buildBPKernels(length: number): void;
    feedForward(input: Layer | Matrix, isInTraining: boolean, gpu?: boolean): any;
    calculate_errors(error: any, input: Matrix): void;
    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix, gpu: boolean): void;
    updateWeights(l_rate: number): void;
    toSavedModel(): SavedLayer;
    fromSavedModel(data: SavedLayer): void;
}
