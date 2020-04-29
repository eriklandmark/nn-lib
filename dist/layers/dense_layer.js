"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const layer_1 = __importDefault(require("./layer"));
const matrix_1 = __importDefault(require("../matrix"));
const sigmoid_1 = __importDefault(require("../activations/sigmoid"));
class DenseLayer extends layer_1.default {
    constructor(layerSize = 1, activation = new sigmoid_1.default()) {
        super();
        this.weights = new matrix_1.default();
        this.errorWeights = new matrix_1.default();
        this.bias = new matrix_1.default();
        this.activationFunction = activation;
        this.layerSize = layerSize;
        this.hasGPUSupport = true;
        this.type = "dense";
    }
    buildLayer(prevLayerShape) {
        this.shape = [this.layerSize];
        this.prevLayerShape = prevLayerShape;
        this.weights = new matrix_1.default();
        this.weights.createEmptyArray(prevLayerShape[0], this.layerSize);
        this.bias = new matrix_1.default();
        this.bias.createEmptyArray(1, this.layerSize);
        this.weights.populateRandom();
        this.bias.populateRandom();
        this.errorWeights = new matrix_1.default();
        this.errorBias = new matrix_1.default();
        this.output_error = new matrix_1.default();
        this.activation = new matrix_1.default();
    }
    buildFFKernels(batch_size) {
        const output_shape = [this.weights.dim().c, batch_size];
        this.ff_kernel = this.gpuInstance.createKernel(function (a, w, b) {
            let sum = 0;
            for (let i = 0; i < this.constants.arr_length; i++) {
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
    }
    buildBPKernels(length) {
        const output_shape = [this.activation.dim().c, this.activation.dim().r];
        this.bp_error_kernel = this.gpuInstance.createKernel(function (a, pW, pO) {
            let sum = 0;
            for (let i = 0; i < this.constants.mmlength; i++) {
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
            let sum = 0;
            for (let i = 0; i < this.constants.arr_length; i++) {
                sum += a[i][this.thread.y] * e[i][this.thread.x];
            }
            return sum;
        })
            .setPrecision("single")
            .setDynamicOutput(true);
        this.bp_error_weight_kernel.immutable = true;
    }
    feedForward(input, isInTraining) {
        if (this.useGpu) {
            const result = this.act_kernel(this.ff_kernel(input, this.weights.toNumberArray(), this.bias.toNumberArray()));
            this.activation = new matrix_1.default(result.toArray());
            return result;
        }
        else {
            let act;
            if (input instanceof matrix_1.default) {
                act = input;
            }
            else {
                act = input.activation;
            }
            const z = act.mm(this.weights);
            z.iterate((i, j) => {
                z.set(i, j, z.get(i, j) + this.bias.get(0, j));
            });
            this.activation = this.activationFunction.normal(z);
        }
    }
    backPropagation(prev_layer, next_layer) {
        if (this.useGpu) {
            let input;
            if (next_layer instanceof layer_1.default) {
                input = next_layer.activation;
            }
            else {
                input = next_layer;
            }
            const error = this.bp_error_kernel(this.activation.toNumberArray(), prev_layer.weights.toNumberArray(), prev_layer.output_error);
            this.output_error = error;
            this.bp_error_weight_kernel.setOutput([this.activation.dim().c, input.dim().c])
                .setConstants({ arr_length: input.dim().r });
            const error_weights = this.bp_error_weight_kernel(input.toNumberArray(), error);
            this.errorWeights = new matrix_1.default(error_weights);
            const errorMatrix = new matrix_1.default(error.toArray());
            this.errorBias = errorMatrix.sum(0);
        }
        else {
            let dzh_dwh;
            if (next_layer instanceof layer_1.default) {
                dzh_dwh = next_layer.activation;
            }
            else {
                dzh_dwh = next_layer;
            }
            const deltaActv = this.activationFunction.derivative(this.activation);
            // @ts-ignore
            const error = (prev_layer.output_error.mm(prev_layer.weights.transpose())).mul(deltaActv);
            this.errorWeights = dzh_dwh.transpose().mm(error);
            this.errorBias = error.sum(0);
            this.output_error = error;
        }
    }
}
exports.default = DenseLayer;
