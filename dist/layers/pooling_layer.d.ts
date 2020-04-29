import Layer from "./layer";
import Tensor from "../tensor";
import { SavedLayer } from "../model";
export default class PoolingLayer extends Layer {
    type: string;
    filterSize: number[];
    padding: number;
    stride: number[];
    channel_first: boolean;
    poolingFunc: string;
    constructor(filterSize?: number[], stride?: number[], ch_first?: boolean);
    buildLayer(prevLayerShape: number[]): void;
    getLayerInfo(): {
        shape: number[];
        type: string;
        activation: string;
    };
    feedForward(input: Layer, isInTraining: boolean): void;
    backPropagation(prev_layer: Layer, next_layer: Layer | Tensor[]): void;
    toSavedModel(): SavedLayer;
    fromSavedModel(data: SavedLayer): void;
}
