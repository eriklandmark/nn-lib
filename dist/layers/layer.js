import { GPU } from "gpu.js";
import Activation from "../activations/activations";
import Tensor from "../tensor";
import Optimizers from "../optimizers/Optimizers";
export default class Layer {
    constructor() {
        this.weights = new Tensor();
        this.bias = new Tensor();
        this.errorWeights = new Tensor();
        this.errorBias = new Tensor();
        this.activation = new Tensor();
        this.useGpu = false;
        this.gpuInstance = new GPU();
        this.shape = [];
        this.prevLayerShape = [];
        this.type = "";
        this.hasGPUSupport = false;
        this.isFirstLayer = false;
    }
    getLayerInfo() {
        return {
            type: this.type,
            shape: this.shape,
            activation: this.activationFunction ? this.activationFunction.name : "NONE"
        };
    }
    buildLayer(prevLayerShape) { }
    feedForward(input, isInTraining) { }
    buildFFKernels(batch_size) { }
    buildBPKernels(size) { }
    backPropagation(prev_layer, next_layer) { }
    toSavedModel() {
        return {
            weights: this.weights.t,
            bias: this.bias.t,
            activation: this.activationFunction ? this.activationFunction.name : "none",
            shape: this.shape,
            prevLayerShape: this.prevLayerShape,
            optimizer: this.optimizer ? this.optimizer.name : "adam",
            layer_specific: {}
        };
    }
    fromSavedModel(data) {
        this.weights = Tensor.fromJsonObject(data.weights);
        this.bias = Tensor.fromJsonObject(data.bias);
        if (data.activation != "none") {
            this.activationFunction = Activation.fromName(data.activation);
        }
        this.shape = data.shape;
        this.prevLayerShape = data.prevLayerShape;
        this.optimizer = new (Optimizers.fromName(data.optimizer))(this);
    }
    updateLayer() {
        this.optimizer.optimizeWeights();
        this.optimizer.optimizeBias();
    }
}
