"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var matrix_1 = __importDefault(require("../../matrix"));
var vector_1 = __importDefault(require("../../vector"));
var gpu_js_1 = require("gpu.js");
var sigmoid_1 = __importDefault(require("../activations/sigmoid"));
var Layer = /** @class */ (function () {
    function Layer(activation) {
        if (activation === void 0) { activation = new sigmoid_1.default(); }
        this.weights = new matrix_1.default();
        this.bias = new vector_1.default();
        this.errorWeights = new matrix_1.default();
        this.errorBias = new matrix_1.default();
        this.output_error = new matrix_1.default();
        this.activation = new matrix_1.default();
        this.useGpu = false;
        this.gpuInstance = new gpu_js_1.GPU();
        this.shape = [];
        this.activationFunction = activation;
    }
    Layer.prototype.setGpuInstance = function (gpuIns) {
        this.gpuInstance = gpuIns;
    };
    Layer.prototype.buildLayer = function (prevLayerShape) { };
    Layer.prototype.feedForward = function (input, isInTraining) { };
    Layer.prototype.backPropagation = function (prev_layer, next_layer) { };
    Layer.prototype.updateWeights = function (l_rate) { };
    return Layer;
}());
exports.default = Layer;
