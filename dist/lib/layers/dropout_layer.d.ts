import Layer from "./layer";
import Matrix from "../../matrix";
import {SavedLayer} from "../../model";

export default class DropoutLayer extends Layer {
    rate: number;
    type: string;
    constructor(rate?: number);
    buildLayer(prevLayerShape: number[]): void;
    feedForward(input: Layer | Matrix, isInTraining: boolean): void;
    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix): void;
    updateWeights(l_rate: number): void;
    toSavedModel(): SavedLayer;
    fromSavedModel(data: SavedLayer): void;
}
