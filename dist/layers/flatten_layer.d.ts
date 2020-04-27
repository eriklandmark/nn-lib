import Layer from "./layer";
import Matrix from "../matrix";
import { SavedLayer } from "../model";
export default class FlattenLayer extends Layer {
    type: string;
    prevShape: number[];
    buildLayer(prevLayerShape: number[]): void;
    feedForward(input: Layer, isInTraining: boolean): void;
    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix): void;
    toSavedModel(): SavedLayer;
    fromSavedModel(data: SavedLayer): void;
}
