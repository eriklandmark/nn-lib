import Layer from "./layer";
import Matrix from "../matrix";
export default class FlattenLayer extends Layer {
    type: string;
    prevShape: number[];
    buildLayer(prevLayerShape: number[]): void;
    feedForward(input: Layer, isInTraining: boolean): void;
    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix): void;
}
