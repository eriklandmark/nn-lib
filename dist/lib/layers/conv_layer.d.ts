import Layer from "./layer";
import Matrix from "../../matrix";

export default class ConvolutionLayer extends Layer {
    filterSize: number[];
    constructor(filterSize: number[]);
    buildLayer(prevLayerShape: number[]): void;
    feedForward(input: Layer | Matrix, isInTraining: boolean): void;
    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix): void;
    updateWeights(l_rate: number): void;
}
