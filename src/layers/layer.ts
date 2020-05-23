import {GPU} from "gpu.js";
import Activation, {IActivation} from "../activations/activations";
import Tensor from "../tensor";
import {SavedLayer} from "../model";
import Optimizers, {IOptimizer} from "../optimizers/Optimizers";

export default class Layer {
    weights: Tensor = new Tensor()
    bias: Tensor = new Tensor()
    errorWeights: Tensor = new Tensor()
    errorBias: Tensor = new Tensor()
    output_error: any
    activation: Tensor = new Tensor()
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
    feedForward(input: Layer | Tensor, isInTraining: boolean) {}
    buildFFKernels(batch_size: number) {}
    buildBPKernels(size: number) {}
    backPropagation(prev_layer: Layer, next_layer: Layer | Tensor) {}

    toSavedModel(): SavedLayer {
        return {
            weights: this.weights.t,
            bias: this.bias.t,
            activation: this.activationFunction? this.activationFunction.name: "none",
            shape: this.shape,
            prevLayerShape: this.prevLayerShape,
            optimizer: this.optimizer? this.optimizer.name : "adam",
            layer_specific: {}
        }
    }

    fromSavedModel(data: SavedLayer) {
        this.weights = Tensor.fromJsonObject(data.weights)
        this.bias = Tensor.fromJsonObject(data.bias)
        if(data.activation != "none") {
            this.activationFunction = Activation.fromName(data.activation)
        }
        this.shape = data.shape
        this.prevLayerShape = data.prevLayerShape
        this.optimizer = new (Optimizers.fromName(data.optimizer))(this)
    }

    updateLayer() {
        this.optimizer.optimizeWeights()
        this.optimizer.optimizeBias()
    }
}