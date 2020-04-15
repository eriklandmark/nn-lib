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
var layer_1 = __importDefault(require("./layer"));
var matrix_1 = __importDefault(require("../../matrix"));
var activations_1 = __importDefault(require("../activations/activations"));
var vector_1 = __importDefault(require("../../vector"));
var sigmoid_1 = __importDefault(require("../activations/sigmoid"));
var DenseLayer = /** @class */ (function (_super) {
    __extends(DenseLayer, _super);
    function DenseLayer(layerSize, activation) {
        if (layerSize === void 0) { layerSize = 1; }
        if (activation === void 0) { activation = new sigmoid_1.default(); }
        var _this = _super.call(this) || this;
        _this.activationFunction = activation;
        _this.layerSize = layerSize;
        _this.type = "dense";
        return _this;
    }
    DenseLayer.prototype.buildLayer = function (prevLayerShape) {
        this.shape = [this.layerSize];
        this.prevLayerShape = prevLayerShape;
        this.weights = new matrix_1.default();
        this.weights.createEmptyArray(prevLayerShape[0], this.layerSize);
        this.bias = new vector_1.default(this.layerSize);
        this.weights.populateRandom();
        this.bias.populateRandom();
        this.errorWeights = new matrix_1.default();
        this.errorBias = new matrix_1.default();
        this.output_error = new matrix_1.default();
        this.activation = new matrix_1.default();
    };
    DenseLayer.prototype.buildFFKernels = function (batch_size) {
        var output_shape = [this.weights.dim().c, batch_size];
        this.ff_kernel = this.gpuInstance.createKernel(function (a, w, b) {
            var sum = 0;
            for (var i = 0; i < this.constants.arr_length; i++) {
                sum += a[this.thread.y][i] * w[i][this.thread.x];
            }
            return sum + b[this.thread.x];
        })
            .setPipeline(true)
            .setPrecision("single")
            .setConstants({ arr_length: this.weights.dim().r })
            .setDynamicOutput(false)
            .setOutput(output_shape);
        this.ff_kernel.immutable = true;
        this.act_kernel = this.gpuInstance.createKernel(this.activationFunction.normal_gpu())
            .setPipeline(true)
            .setConstants({ softmax: this.weights.dim().c })
            .setPrecision("single")
            .setDynamicOutput(false)
            .setOutput(output_shape);
        this.act_kernel.immutable = true;
    };
    DenseLayer.prototype.buildBPKernels = function (length) {
        var output_shape = [this.activation.dim().c, this.activation.dim().r];
        this.bp_error_kernel = this.gpuInstance.createKernel(function (a, pW, pO) {
            var sum = 0;
            for (var i = 0; i < this.constants.mmlength; i++) {
                sum += pO[this.thread.y][i] * pW[this.thread.x][i];
            }
            // @ts-ignore
            return sum * actv_der(a[this.thread.y][this.thread.x]);
        })
            .addFunction(this.activationFunction.derivative_gpu(), { output: output_shape })
            .setPipeline(true)
            .setPrecision("single")
            .setDynamicOutput(false)
            .setOutput(output_shape)
            .setConstants({ mmlength: length });
        this.bp_error_kernel.immutable = true;
        this.bp_error_weight_kernel = this.gpuInstance.createKernel(function (a, e) {
            var sum = 0;
            for (var i = 0; i < this.constants.arr_length; i++) {
                sum += a[i][this.thread.y] * e[i][this.thread.x];
            }
            return sum;
        })
            .setPrecision("single")
            .setDynamicOutput(true);
        this.bp_error_weight_kernel.immutable = true;
    };
    DenseLayer.prototype.feedForward = function (input, isInTraining, gpu) {
        var _this = this;
        if (gpu === void 0) { gpu = false; }
        if (gpu) {
            var result = this.act_kernel(this.ff_kernel(input, this.weights.toNumberArray(), this.bias.toNumberArray()));
            this.activation = new matrix_1.default(result.toArray());
            return result;
        }
        else {
            var act = void 0;
            if (input instanceof matrix_1.default) {
                act = input;
            }
            else {
                act = input.activation;
            }
            var z_1 = act.mm(this.weights);
            //console.log(z.toString(10, 6))
            z_1.iterate(function (i, j) {
                z_1.set(i, j, z_1.get(i, j) + _this.bias.get(j));
            });
            this.activation = this.activationFunction.normal(z_1);
            //console.log(this.activation.toString())
        }
    };
    DenseLayer.prototype.calculate_errors = function (error, input) {
    };
    DenseLayer.prototype.backPropagation = function (prev_layer, next_layer, gpu) {
        if (gpu) {
            var input = void 0;
            if (next_layer instanceof layer_1.default) {
                input = next_layer.activation;
            }
            else {
                input = next_layer;
            }
            var error = this.bp_error_kernel(this.activation.toNumberArray(), prev_layer.weights.toNumberArray(), prev_layer.output_error);
            this.output_error = error;
            this.bp_error_weight_kernel.setOutput([this.activation.dim().c, input.dim().c])
                .setConstants({ arr_length: input.dim().r });
            var error_weights = this.bp_error_weight_kernel(input.toNumberArray(), error);
            this.errorWeights = new matrix_1.default(error_weights);
            var errorMatrix = new matrix_1.default(error.toArray());
            this.errorBias = errorMatrix.sum(0);
        }
        else {
            var dzh_dwh = void 0;
            if (next_layer instanceof layer_1.default) {
                dzh_dwh = next_layer.activation;
            }
            else {
                dzh_dwh = next_layer;
            }
            var deltaActv = this.activationFunction.derivative(this.activation);
            // @ts-ignore
            var error = (prev_layer.output_error.mm(prev_layer.weights.transpose())).mul(deltaActv);
            this.errorWeights = dzh_dwh.transpose().mm(error);
            this.errorBias = error.sum(0);
            this.output_error = error;
        }
    };
    DenseLayer.prototype.updateWeights = function (l_rate) {
        var _this = this;
        this.weights = this.weights.sub(this.errorWeights.mul(l_rate));
        this.bias.iterate(function (val, i) {
            _this.bias.set(i, val - (_this.errorBias.get(0, i) * l_rate));
        });
    };
    DenseLayer.prototype.toSavedModel = function () {
        return {
            weights: this.weights.matrix,
            bias: this.bias.vector,
            shape: this.shape,
            activation: this.activationFunction.name
        };
    };
    DenseLayer.prototype.fromSavedModel = function (data) {
        this.weights = matrix_1.default.fromJsonObject(data.weights);
        this.bias = vector_1.default.fromJsonObj(data.bias);
        this.shape = data.shape;
        this.activationFunction = activations_1.default.fromName(data.activation);
    };
    return DenseLayer;
}(layer_1.default));
exports.default = DenseLayer;
