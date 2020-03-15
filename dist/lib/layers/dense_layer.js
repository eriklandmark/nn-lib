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
            act = input;
        }
        else {
            act = input.activation;
        }
        if (this.useGpu) {
            /*
            const ffKernel = this.gpuInstance.createKernelMap({
                addResult: Matrix.addGpu(),
                multiplyResult: Matrix.mmGpu(),
                actvResult: this.activationFunction.normal_gpu()
            }, function (a: ThreadKernelVariable, b: ThreadKernelVariable, c: ThreadKernelVariable) {
                //@ts-ignore
                return actv(add(mm(a, b), c[this.thread.x]));
            }, {output: [this.weights.dim().c, act.dim().r], constants: {mmLength: act.dim().c}})
            ffKernel.setLoopMaxIterations(Math.max(act.dim().c, this.weights.dim().r))
            this.activation = new Matrix(<Float32Array[]>ffKernel(act.toNumberArray(), this.weights.toNumberArray(), this.bias.toNumberArray())["result"]);
            ffKernel.destroy()*/
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
        var deltaActv = this.activationFunction.derivative(this.activation);
        // @ts-ignore
        var error = (prev_layer.output_error.mm(prev_layer.weights.transpose())).mul(deltaActv);
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
