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
var vector_1 = __importDefault(require("../../vector"));
var DenseLayer = /** @class */ (function (_super) {
    __extends(DenseLayer, _super);
    function DenseLayer(layerSize, activation) {
        var _this = _super.call(this, activation) || this;
        _this.layerSize = layerSize;
        return _this;
    }
    DenseLayer.prototype.buildLayer = function (prevLayerShape) {
        this.shape = [this.layerSize];
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
    DenseLayer.prototype.feedForward = function (input, isInTraining) {
        var _this = this;
        var act;
        if (input instanceof matrix_1.default) {
            if (this.activation.empty()) {
                this.activation.createEmptyArray(input.dim().r, this.layerSize);
            }
            act = input;
        }
        else {
            if (this.activation.empty()) {
                this.activation.createEmptyArray(input.activation.dim().r, this.layerSize);
            }
            act = input.activation;
        }
        if (this.useGpu) {
            var ffKernel = this.gpuInstance.createKernelMap({
                addResult: matrix_1.default.addGpu(),
                multiplyResult: matrix_1.default.mmGpu(),
                actvResult: this.activationFunction.normal_gpu()
            }, function (a, b, c) {
                //@ts-ignore
                return actv(add(mm(a, b), c[this.thread.x]));
            }, { output: [this.weights.dim().c, act.dim().r], constants: { mmLength: act.dim().c } });
            ffKernel.setLoopMaxIterations(Math.max(act.dim().c, this.weights.dim().r));
            this.activation = new matrix_1.default(ffKernel(act.toNumberArray(), this.weights.toNumberArray(), this.bias.toNumberArray())["result"]);
            ffKernel.destroy();
        }
        else {
            var z_1 = act.mm(this.weights);
            z_1.iterate(function (i, j) {
                z_1.set(i, j, z_1.get(i, j) + _this.bias.get(j));
            });
            this.activation = this.activationFunction.normal(z_1);
        }
    };
    DenseLayer.prototype.backPropagation = function (prev_layer, next_layer) {
        var dzh_dwh;
        if (next_layer instanceof layer_1.default) {
            dzh_dwh = next_layer.activation;
        }
        else {
            dzh_dwh = next_layer;
        }
        /*
        const feedForwardKernel = gpu.createKernelMap({
            addResult: Matrix.addGpu(),
            multiplyResult: Matrix.mmGpu(),
            actvResult: Activations.sigmoid_gpu()
        }, function(a, b, c) {
            //@ts-ignore
            return actv(add(mm(a, b), c[this.thread.y][this.thread.x]));
        }, { output: [b.dim().c, a.dim().r], constants: {mmLength: a.dim().c}})
        feedForwardKernel.setLoopMaxIterations(Math.max(a.dim().c, b.dim().r))


        new Matrix(<Float32Array[]>feedForwardKernel(a.toNumberArray(), b.toNumberArray(), c.toNumberArray()).result)
        */
        var error = prev_layer.output_error.mm(prev_layer.weights.transpose()).mul(this.activationFunction.derivative(this.activation));
        this.errorWeights = dzh_dwh.transpose().mm(error);
        this.errorBias = error.sum(0);
        this.output_error = error;
    };
    DenseLayer.prototype.updateWeights = function (l_rate) {
        var _this = this;
        this.weights = this.weights.sub(this.errorWeights.mul(l_rate));
        this.bias.iterate(function (val, i) {
            _this.bias.set(i, val - (_this.errorBias.get(0, i) * l_rate));
        });
    };
    return DenseLayer;
}(layer_1.default));
exports.default = DenseLayer;
