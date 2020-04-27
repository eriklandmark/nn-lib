import Layer from "./layer";
import Tensor from "../tensor";
import { SavedLayer } from "../model";
export default class PoolingLayer extends Layer {
    type: string;
    prevShape: number[];
    filterSize: number[];
    padding: number;
    stride: number[];
    channel_first: boolean;
    poolingFunc: string;
    constructor(filterSize?: number[], stride?: number[], ch_first?: boolean);
    buildLayer(prevLayerShape: number[]): void;
    feedForward(input: Layer, isInTraining: boolean): void;
    backPropagation(prev_layer: Layer, next_layer: Layer | Tensor[]): void;
    toSavedModel(): SavedLayer;
    fromSavedModel(data: SavedLayer): void;
}
