import Layer from "./layer";
import { SavedLayer } from "../model";
export default class DropoutLayer extends Layer {
    rate: number;
    type: string;
    constructor(rate?: number);
    buildLayer(prevLayerShape: number[]): void;
    feedForward(input: Layer, isInTraining: boolean): void;
    backPropagation(prev_layer: Layer, next_layer: Layer): void;
    updateLayer(): void;
    toSavedModel(): SavedLayer;
    fromSavedModel(data: SavedLayer): void;
}
