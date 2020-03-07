"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var matrix_1 = __importDefault(require("../../matrix"));
var vector_1 = __importDefault(require("../../vector"));
var activations_1 = __importDefault(require("../activations"));
var Layer = /** @class */ (function () {
    function Layer(layerSize, activation) {
        if (activation === void 0) { activation = "sigmoid"; }
        this.useGpu = false;
        this.layerSize = layerSize;
        this.activationString = activation;
    }
    Layer.prototype.buildLayer = function (prevLayerSize) {
        this.weights = new matrix_1.default();
        this.weights.createEmptyArray(prevLayerSize, this.layerSize);
        this.bias = new vector_1.default(this.layerSize);
        this.weights.populateRandom();
        this.bias.populateRandom();
        this.errorWeights = new matrix_1.default();
        this.errorBias = new matrix_1.default();
        this.output_error = new matrix_1.default();
        this.activation = new matrix_1.default();
        var _a = activations_1.default.lookUp(this.activationString), func = _a.func, derv = _a.derv;
        this.actFunc = func;
        this.actFuncDer = derv;
    };
    Layer.prototype.setGpuInstance = function (gpuIns) {
        this.gpuInstance = gpuIns;
    };
    Layer.prototype.feedForward = function (input) {
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
                actvResult: activations_1.default.sigmoid_gpu()
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
            this.activation = this.actFunc(z_1);
        }
    };
    Layer.prototype.updateWeights = function (l_rate) {
        var _this = this;
        this.weights = this.weights.sub(this.errorWeights.mul(l_rate));
        this.bias.iterate(function (val, i) {
            _this.bias.set(i, val - (_this.errorBias.get(0, i) * l_rate));
        });
    };
    return Layer;
}());
exports.default = Layer;
