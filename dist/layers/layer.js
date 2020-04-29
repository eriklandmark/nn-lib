"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const matrix_1 = __importDefault(require("../matrix"));
const vector_1 = __importDefault(require("../vector"));
const gpu_js_1 = require("gpu.js");
const activations_1 = __importDefault(require("../activations/activations"));
const tensor_1 = __importDefault(require("../tensor"));
const Optimizers_1 = __importDefault(require("../optimizers/Optimizers"));
class Layer {
    constructor() {
        this.weights = new matrix_1.default();
        this.bias = new matrix_1.default();
        this.errorWeights = new matrix_1.default();
        this.errorBias = new matrix_1.default();
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
            weights: this.weights instanceof matrix_1.default ? this.weights.matrix : this.weights.map((t) => t.tensor),
            bias: this.bias instanceof vector_1.default ? this.bias.vector : this.bias.matrix,
            activation: this.activationFunction ? this.activationFunction.name : "none",
            shape: this.shape,
            prevLayerShape: this.prevLayerShape,
            optimizer: this.optimizer.name,
            layer_specific: {}
        };
    }
    fromSavedModel(data) {
        this.weights = this.weights instanceof matrix_1.default ? matrix_1.default.fromJsonObject(data.weights) :
            data.weights.map((t) => tensor_1.default.fromJsonObject(t));
        this.bias = matrix_1.default.fromJsonObject(data.bias);
        if (data.activation != "none") {
            this.activationFunction = activations_1.default.fromName(data.activation);
        }
        this.shape = data.shape;
        this.prevLayerShape = data.prevLayerShape;
        const opt = Optimizers_1.default.fromName(data.optimizer);
        this.optimizer = new opt(this);
    }
    updateLayer() {
        this.optimizer.optimizeWeights();
        this.optimizer.optimizeBias();
    }
}
exports.default = Layer;
