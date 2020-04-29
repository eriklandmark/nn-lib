import Matrix from "../matrix";
import Vector from "../vector";
import {GPU} from "gpu.js";
import Activation, {IActivation} from "../activations/activations";
import Tensor from "../tensor";
import {SavedLayer} from "../model";
import Optimizers, {IOptimizer} from "../optimizers/Optimizers";

export default class Layer {
    weights: Matrix | Tensor[] = new Matrix()
    bias: Matrix = new Matrix()
    errorWeights: Matrix | Tensor[] = new Matrix()
    errorBias: Matrix = new Matrix()
    output_error: any
    activation: Matrix | Tensor[]
    activationFunction: IActivation
    useGpu: boolean = false;
    gpuInstance: GPU = new GPU()
    shape: number[] = []
    prevLayerShape: number[] = []
    type: string = ""
    hasGPUSupport = false;
    isFirstLayer = false;
    optimizer: IOptimizer
    learning_rate: number

    ff_kernel: any
    act_kernel: any
    bp_error_kernel: any
    bp_error_weight_kernel: any

    getLayerInfo() {
        return {
            type: this.type,
            shape: this.shape,
            activation: this.activationFunction ? this.activationFunction.name : "NONE"
        }
    }

    buildLayer(prevLayerShape: number[]) {}
    feedForward(input: Layer | Matrix | Tensor[], isInTraining: boolean) {}
    buildFFKernels(batch_size: number) {}
    buildBPKernels(size: number) {}
    backPropagation(prev_layer: Layer, next_layer: Layer | Matrix | Tensor[]) {}

    toSavedModel(): SavedLayer {
        return {
            weights: this.weights instanceof Matrix? this.weights.matrix: this.weights.map((t) => t.tensor),
            bias: this.bias instanceof Vector? this.bias.vector : this.bias.matrix,
            activation: this.activationFunction? this.activationFunction.name: "none",
            shape: this.shape,
            prevLayerShape: this.prevLayerShape,
            optimizer: this.optimizer.name,
            layer_specific: {}
        }
    }

    fromSavedModel(data: SavedLayer) {
        this.weights = this.weights instanceof Matrix? Matrix.fromJsonObject(data.weights):
            (<Float32Array[][][]>data.weights).map((t) => Tensor.fromJsonObject(t))
        this.bias = Matrix.fromJsonObject(<Float32Array[]>data.bias)
        if(data.activation != "none") {
            this.activationFunction = Activation.fromName(data.activation)
        }
        this.shape = data.shape
        this.prevLayerShape = data.prevLayerShape
        const opt = Optimizers.fromName(data.optimizer)
        this.optimizer = new opt(this)
    }

    updateLayer() {
        this.optimizer.optimizeWeights()
        this.optimizer.optimizeBias()
    }
}