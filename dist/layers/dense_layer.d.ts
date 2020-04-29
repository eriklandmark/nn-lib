import Layer from "./layer";
import Matrix from "../matrix";
import { IActivation } from "../activations/activations";
export default class DenseLayer extends Layer {
    layerSize: number;
    ff_kernel: any;
    act_kernel: any;
    bp_error_kernel: any;
    bp_error_weight_kernel: any;
    weights: Matrix;
    errorWeights: Matrix;
    bias: Matrix;
    constructor(layerSize?: number, activation?: IActivation);
    buildLayer(prevLayerShape: number[]): void;
    buildFFKernels(batch_size: number): void;
    buildBPKernels(length: number): void;
    feedForward(input: Layer | Matrix, isInTraining: boolean): any;
    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix): void;
}
