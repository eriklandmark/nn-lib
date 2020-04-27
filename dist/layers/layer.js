"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const matrix_1 = __importDefault(require("../matrix"));
const vector_1 = __importDefault(require("../vector"));
const gpu_js_1 = require("gpu.js");
class Layer {
    constructor() {
        this.weights = new matrix_1.default();
        this.bias = new vector_1.default();
        this.errorWeights = new matrix_1.default();
        this.errorBias = new matrix_1.default();
        this.activation = new matrix_1.default();
        this.useGpu = false;
        this.gpuInstance = new gpu_js_1.GPU();
        this.shape = [];
        this.prevLayerShape = [];
        this.type = "";
        this.hasGPUSupport = false;
        this.isFirstLayer = false;
    }
    setGpuInstance(gpuIns) {
        this.gpuInstance = gpuIns;
    }
    getLayerInfo() {
        return {
            type: this.type,
            shape: this.shape,
            activation: this.activationFunction ? this.activationFunction.name : "NO ACTIVATION"
        };
    }
    buildLayer(prevLayerShape) { }
    feedForward(input, isInTraining) { }
    buildFFKernels(batch_size) { }
    buildBPKernels(size) { }
    backPropagation(prev_layer, next_layer) { }
    calculate_errors(error, next_layer) { }
    updateWeights(l_rate) { }
    toSavedModel() { return; }
    fromSavedModel(data) { }
}
exports.default = Layer;
