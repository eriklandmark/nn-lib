"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var matrix_1 = __importDefault(require("../../matrix"));
var dense_layer_1 = __importDefault(require("./dense_layer"));
var activations_1 = __importDefault(require("../activations/activations"));
var losses_1 = __importDefault(require("../losses/losses"));
var mean_squared_error_1 = __importDefault(require("../losses/mean_squared_error"));
var vector_1 = __importDefault(require("../../vector"));
var sigmoid_1 = __importDefault(require("../activations/sigmoid"));
var OutputLayer = /** @class */ (function (_super) {
    __extends(OutputLayer, _super);
    function OutputLayer(layerSize, activation) {
        if (layerSize === void 0) { layerSize = 1; }
        if (activation === void 0) { activation = new sigmoid_1.default(); }
        var _this = _super.call(this, layerSize, activation) || this;
        _this.loss = 0;
        _this.layerSize = 0;
        _this.lossFunction = new mean_squared_error_1.default();
        _this.layerSize = layerSize;
        _this.type = "output";
        return _this;
    }
    OutputLayer.prototype.backPropagationOutputLayer = function (labels, next_layer) {
        this.loss = labels.mul(-1).mul(this.activation.log()).sum();
        var nextActv = next_layer.activation.transpose();
        var gradient = this.lossFunction.derivative(this.activation, labels);
        this.errorBias = gradient;
        this.output_error = gradient;
        if (this.useGpu) {
            var errorWeightsKernel = this.gpuInstance.createKernel(matrix_1.default.mmGpu())
                .setOutput([labels.dim().c, nextActv.dim().r]).setConstants({ mmLength: labels.dim().r });
            errorWeightsKernel.setLoopMaxIterations(Math.max(this.activation.dim().r, nextActv.dim().c));
            this.errorWeights = new matrix_1.default(errorWeightsKernel(nextActv.toNumberArray(), gradient.toNumberArray()));
            errorWeightsKernel.destroy();
        }
        else {
            this.errorWeights = next_layer.activation.transpose().mm(gradient);
        }
    };
    OutputLayer.prototype.toSavedModel = function () {
        return {
            weights: this.weights.matrix,
            bias: this.bias.vector,
            loss: this.lossFunction.name,
            shape: this.shape,
            activation: this.activationFunction.name
        };
    };
    OutputLayer.prototype.fromSavedModel = function (data) {
        this.weights = matrix_1.default.fromJsonObject(data.weights);
        this.bias = vector_1.default.fromJsonObj(data.bias);
        this.shape = data.shape;
        this.activationFunction = activations_1.default.fromName(data.activation);
        this.lossFunction = losses_1.default.fromName(data.loss);
    };
    return OutputLayer;
}(dense_layer_1.default));
exports.default = OutputLayer;
