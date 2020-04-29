import Layer from "./layer";
import Matrix from "../matrix";
export default class BatchNormLayer extends Layer {
    momentum: number;
    running_mean: Matrix;
    running_var: Matrix;
    cache: any;
    weights: Matrix;
    errorWeights: Matrix;
    constructor(momentum?: number);
    buildLayer(prevLayerShape: number[]): void;
    feedForward(input: Layer | Matrix, isInTraining: boolean): void;
    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix): void;
}
