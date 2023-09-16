import Layer from "./layer";
import Tensor from "../tensor";
import { SavedLayer } from "../model";
export default class PoolingLayer extends Layer {
    type: string;
    filterSize: number[];
    padding: number;
    stride: number[];
    channel_first: boolean;
    poolingFuncName: "max" | "avg";
    poolingFunc: Function;
    constructor(filterSize?: number[], stride?: number[], poolingFuncName?: "max" | "avg", ch_first?: boolean);
    buildLayer(prevLayerShape: number[]): void;
    calcPoolFunc(): void;
    getLayerInfo(): {
        shape: number[];
        type: string;
        activation: string;
    };
    feedForward(input: Layer, isInTraining: boolean): void;
    backPropagation(prev_layer: Layer, next_layer: Layer | Tensor): void;
    updateLayer(): void;
    toSavedModel(): SavedLayer;
    fromSavedModel(data: SavedLayer): void;
}
