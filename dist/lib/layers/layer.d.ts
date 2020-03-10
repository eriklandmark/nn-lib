import Matrix from "../../matrix";
import Vector from "../../vector";
import {GPU} from "gpu.js";
import IActivation from "../activations/activations";

export default class Layer {
    weights: Matrix;
    bias: Vector;
    errorWeights: Matrix;
    errorBias: Matrix;
    output_error: Matrix;
    activation: Matrix;
    activationFunction: IActivation;
    useGpu: boolean;
    gpuInstance: GPU;
    shape: number[];
    constructor(activation?: IActivation);
    setGpuInstance(gpuIns: GPU): void;
    buildLayer(prevLayerShape: number[]): void;
    feedForward(input: Layer | Matrix, isInTraining: boolean): void;
    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix): void;
    updateWeights(l_rate: number): void;
}
