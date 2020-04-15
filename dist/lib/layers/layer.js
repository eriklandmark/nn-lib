"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var matrix_1 = __importDefault(require("../../matrix"));
var vector_1 = __importDefault(require("../../vector"));
var gpu_js_1 = require("gpu.js");
var Layer = /** @class */ (function () {
    function Layer() {
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
    }
    Layer.prototype.setGpuInstance = function (gpuIns) {
        this.gpuInstance = gpuIns;
    };
    Layer.prototype.getLayerInfo = function () {
        return {
            type: this.type,
            shape: this.shape,
            activation: this.activationFunction ? this.activationFunction.name : "NO ACTIVATION"
        };
    };
    Layer.prototype.buildLayer = function (prevLayerShape) { };
    Layer.prototype.feedForward = function (input, isInTraining, gpu) {
        if (gpu === void 0) { gpu = false; }
    };
    Layer.prototype.buildFFKernels = function (batch_size) { };
    Layer.prototype.buildBPKernels = function (size) { };
    Layer.prototype.backPropagation = function (prev_layer, next_layer, gpu) {
        if (gpu === void 0) { gpu = false; }
    };
    Layer.prototype.calculate_errors = function (error, next_layer) { };
    Layer.prototype.updateWeights = function (l_rate) { };
    Layer.prototype.toSavedModel = function () { return; };
    Layer.prototype.fromSavedModel = function (data) { };
    return Layer;
}());
exports.default = Layer;
