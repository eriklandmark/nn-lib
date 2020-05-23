"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const gpu_js_1 = require("gpu.js");
const activations_1 = __importDefault(require("../activations/activations"));
const tensor_1 = __importDefault(require("../tensor"));
const Optimizers_1 = __importDefault(require("../optimizers/Optimizers"));
class Layer {
    constructor() {
        this.weights = new tensor_1.default();
        this.bias = new tensor_1.default();
        this.errorWeights = new tensor_1.default();
        this.errorBias = new tensor_1.default();
        this.activation = new tensor_1.default();
        this.useGpu = false;
        this.gpuInstance = new gpu_js_1.GPU();
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
        this.weights = tensor_1.default.fromJsonObject(data.weights);
        this.bias = tensor_1.default.fromJsonObject(data.bias);
        if (data.activation != "none") {
            this.activationFunction = activations_1.default.fromName(data.activation);
        }
        this.shape = data.shape;
        this.prevLayerShape = data.prevLayerShape;
        this.optimizer = new (Optimizers_1.default.fromName(data.optimizer))(this);
    }
    updateLayer() {
        this.optimizer.optimizeWeights();
        this.optimizer.optimizeBias();
    }
}
exports.default = Layer;
