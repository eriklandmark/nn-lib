"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const layer_1 = __importDefault(require("./layer"));
const matrix_1 = __importDefault(require("../matrix"));
class BatchNormLayer extends layer_1.default {
    constructor(momentum = 0.9) {
        super();
        this.cache = {};
        this.momentum = momentum;
        this.type = "batch_norm";
    }
    buildLayer(prevLayerShape) {
        const [D] = prevLayerShape;
        console.log(prevLayerShape);
        this.shape = prevLayerShape;
        this.running_mean = new matrix_1.default();
        this.running_mean.createEmptyArray(1, D);
        this.running_var = new matrix_1.default();
        this.running_var.createEmptyArray(1, D);
        this.weights = new matrix_1.default();
        this.weights.createEmptyArray(1, D);
        this.weights.populateRandom();
        this.bias = new matrix_1.default();
        this.bias.createEmptyArray(1, D);
        this.bias.populateRandom();
    }
    feedForward(input, isInTraining) {
        let act;
        if (input instanceof matrix_1.default) {
            act = input;
        }
        else {
            act = input.activation;
        }
        const N = act.dim().r;
        const mean = act.mean(0);
        const diff = act.sub(mean.repeat(0, N));
        const variance = diff.pow(2).mean(0);
        if (isInTraining) {
            console.log(act.toString());
            const xhat = diff.div(variance.sqrt().repeat(0, N).add(Math.pow(10, -5)));
            this.activation = this.weights.repeat(0, N).mul(xhat).add(this.bias.repeat(0, N));
            this.running_mean = this.running_mean.mul(this.momentum).add(mean.mul(1 - this.momentum));
            this.running_var = this.running_var.mul(this.momentum).add(variance.mul(1 - this.momentum));
            this.cache = { variance, diff, xhat };
        }
    }
    backPropagation(prev_layer, next_layer) {
        let error;
        if (prev_layer.output_error instanceof matrix_1.default) {
            error = prev_layer.output_error;
        }
        else {
            error = new matrix_1.default(prev_layer.output_error.toArray());
        }
        let X;
        if (next_layer instanceof matrix_1.default) {
            X = next_layer;
        }
        else {
            X = next_layer.activation;
        }
        const { variance, diff, xhat } = this.cache;
        const dout = error.mm(prev_layer.weights.transpose());
        const N = dout.dim().r;
        const std_inv = variance.sqrt().inv_el(Math.pow(10, -8));
        const dX_norm = dout.mul(this.weights.repeat(0, N));
        const dVar = dX_norm.mul(diff).sum(0).mul(-0.5).mul(std_inv.pow(3));
        const dMean = dX_norm.mul(std_inv.mul(-1).repeat(0, N)).sum(0).add(dVar.mul(diff.mul(-2).mean(0)));
        this.output_error = dX_norm.mul(std_inv.repeat(0, N)).add(dVar.repeat(0, N).mul(2).mul(diff).div(N)).add(dMean.div(N).repeat(0, N));
        this.errorWeights = dout.mul(xhat).sum(0);
        this.errorBias = dout.sum(0);
    }
}
exports.default = BatchNormLayer;
