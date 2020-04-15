import Matrix from "../../matrix";
import Vector from "../../vector";
import { GPU } from "gpu.js";
import { IActivation } from "../activations/activations";
import Tensor from "../../tensor";
import { SavedLayer } from "../../model";
export default class Layer {
    weights: Matrix;
    bias: Vector;
    errorWeights: Matrix;
    errorBias: Matrix | Vector;
    output_error: any;
    activation: Matrix | Tensor[];
    activationFunction: IActivation;
    useGpu: boolean;
    gpuInstance: GPU;
    shape: number[];
    prevLayerShape: number[];
    type: string;
    ff_kernel: any;
    act_kernel: any;
    bp_error_kernel: any;
    bp_error_weight_kernel: any;
    setGpuInstance(gpuIns: GPU): void;
    getLayerInfo(): {
        type: string;
        shape: number[];
        activation: string;
    };
    buildLayer(prevLayerShape: number[]): void;
    feedForward(input: Layer | Matrix | Tensor[], isInTraining: boolean, gpu?: boolean): void;
    buildFFKernels(batch_size: number): void;
    buildBPKernels(size: number): void;
    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix | Tensor[], gpu?: boolean): void;
    calculate_errors(error: any, next_layer: Layer | Matrix): void;
    updateWeights(l_rate: number): void;
    toSavedModel(): SavedLayer;
    fromSavedModel(data: SavedLayer): void;
}
