import Matrix from "../../matrix";
import Vector from "../../vector";
import {GPU} from "gpu.js";

export default class Layer {
    weights: Matrix = new Matrix()
    bias: Vector = new Vector()
    errorWeights: Matrix = new Matrix()
    errorBias: Matrix = new Matrix()
    output_error: Matrix = new Matrix()
    activation: Matrix = new Matrix()
    activationString: string
    actFunc: Function = () => {}
    actFuncDer: Function | null = () => {}
    useGpu: boolean = false;
    gpuInstance: GPU = new GPU()
    shape: number[] = []

    constructor(activation: string = "sigmoid") {
        this.activationString = activation
    }

    setGpuInstance(gpuIns: GPU) {
        this.gpuInstance = gpuIns;
    }

    buildLayer(prevLayerShape: number[]) {}
    feedForward(input: Layer | Matrix, isInTraining: boolean) {}
    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix) {}

    updateWeights(l_rate: number) {
        this.weights = this.weights.sub(this.errorWeights.mul(l_rate))
        this.bias.iterate((val: number, i: number) => {
            this.bias.set(i, val - (this.errorBias.get(0, i) * l_rate))
        })
    }
}