import Matrix from "../../matrix";
import Vector from "../../vector";
import {GPU} from "gpu.js";
import IActivation from "../activations/activations";
import Sigmoid from "../activations/sigmoid";
import Tensor from "../../tensor";

export default class Layer {
    weights: Matrix = new Matrix()
    bias: Vector = new Vector()
    errorWeights: Matrix = new Matrix()
    errorBias: Matrix = new Matrix()
    output_error: Matrix = new Matrix()
    activation: Matrix | Tensor[] = new Matrix()
    activationFunction: IActivation
    useGpu: boolean = false;
    gpuInstance: GPU = new GPU()
    shape: number[] = []

    constructor(activation: IActivation = new Sigmoid()) {
        this.activationFunction = activation
    }

    setGpuInstance(gpuIns: GPU) {
        this.gpuInstance = gpuIns;
    }

    buildLayer(prevLayerShape: number[]) {}
    feedForward(input: Layer | Matrix | Tensor[], isInTraining: boolean) {}
    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix) {}

    updateWeights(l_rate: number) {}
}