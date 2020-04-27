import Layer from "./layer";
import Tensor from "../tensor";
import { IActivation } from "../activations/activations";
import { SavedLayer } from "../model";
export default class ConvolutionLayer extends Layer {
    filterSize: number[];
    filters: Tensor[];
    padding: number;
    stride: number;
    nr_filters: number;
    errorFilters: Tensor[];
    errorInput: Tensor[];
    channel_first: boolean;
    ff_kernel: any;
    act_kernel: any;
    bp_error_kernel: any;
    bp_error_weight_kernel: any;
    useMM: boolean;
    constructor(nr_filters: number, filterSize: number[], ch_first: boolean, activation: IActivation);
    buildLayer(prevLayerShape: number[]): void;
    feedForward(input: Layer | Tensor[], isInTraining: boolean): void;
    buildFFKernels(batch_size: number): void;
    backPropagation(prev_layer: Layer, next_layer: Layer | Tensor[]): void;
    convolve(image: Tensor, filters: Tensor[], channel_first?: boolean): Tensor | Tensor[];
    updateWeights(l_rate: number): void;
    toSavedModel(): SavedLayer;
    fromSavedModel(data: SavedLayer): void;
}