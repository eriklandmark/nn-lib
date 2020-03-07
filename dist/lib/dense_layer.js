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
var DenseLayer = /** @class */ (function (_super) {
    __extends(DenseLayer, _super);
    function DenseLayer() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
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
        var error = prev_layer.output_error.mm(prev_layer.weights.transpose()).mul(this.actFuncDer(this.activation));
        this.errorWeights = dzh_dwh.transpose().mm(error);
        this.errorBias = error.sum(0);
        this.output_error = error;
    };
    return DenseLayer;
}(layer_1.default));
exports.default = DenseLayer;
