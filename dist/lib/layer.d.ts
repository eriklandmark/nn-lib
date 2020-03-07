import Matrix from "./matrix";
import Vector from "./vector";
import {GPU} from "gpu.js";

export default class Layer {
    weights: Matrix;
    bias: Vector;
    errorWeights: Matrix;
    errorBias: Matrix;
    output_error: Matrix;
    activation: Matrix;
    activationString: string;
    actFunc: Function;
    actFuncDer: Function;
    layerSize: number;
    useGpu: boolean;
    gpuInstance: GPU;
    constructor(layerSize: number, activation?: string);
    buildLayer(prevLayerSize: number): void;
    setGpuInstance(gpuIns: GPU): void;
    feedForward(input: Layer | Matrix): void;
    updateWeights(l_rate: number): void;
}
