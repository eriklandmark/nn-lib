import Layer from "./layer";
import Tensor from "../tensor";
export default class FlattenLayer extends Layer {
    type: string;
    prevShape: number[];
    buildLayer(prevLayerShape: number[]): void;
    feedForward(input: Layer, isInTraining: boolean): void;
    backPropagation(prev_layer: Layer, next_layer: Layer | Tensor): void;
    updateLayer(): void;
}
