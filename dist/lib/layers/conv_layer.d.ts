import Layer from "./layer";
import Tensor from "../../tensor";
import {IActivation} from "../activations/activations";
import {SavedLayer} from "../../model";

export default class ConvolutionLayer extends Layer {
    filterSize: number[];
    filters: Tensor[];
    prevLayerShape: number[];
    padding: number;
    stride: number;
    nr_filters: number;
    errorFilters: Tensor[];
    errorInput: Tensor[];
    constructor(nr_filters?: number, filterSize?: number[], activation?: IActivation);
    buildLayer(prevLayerShape: number[]): void;
    feedForward(input: Layer | Tensor[], isInTraining: boolean): void;
    backPropagation(prev_layer: Layer, next_layer: Layer | Tensor[]): void;
    updateWeights(l_rate: number): void;
    toSavedModel(): SavedLayer;
    fromSavedModel(data: SavedLayer): void;
}
