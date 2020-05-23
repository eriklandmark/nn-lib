"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const layer_1 = __importDefault(require("./layer"));
const tensor_1 = __importDefault(require("../tensor"));
class FlattenLayer extends layer_1.default {
    constructor() {
        super(...arguments);
        this.type = "flatten";
        this.prevShape = [];
    }
    buildLayer(prevLayerShape) {
        this.prevShape = prevLayerShape;
        this.shape = [prevLayerShape.reduce((acc, val) => acc * val)];
    }
    feedForward(input, isInTraining) {
        this.activation = new tensor_1.default(input.activation.t.map((t) => new tensor_1.default(t).vectorize(true).t));
    }
    backPropagation(prev_layer, next_layer) {
        const dout = prev_layer.output_error.dot(prev_layer.weights.transpose());
        this.output_error = new tensor_1.default([prev_layer.output_error.shape[0], this.prevShape[0], this.prevShape[1], this.prevShape[2]], true);
        let [h, w, d] = this.prevShape;
        dout.iterate((n, i) => {
            const r = Math.floor(i / (w * d));
            const c = Math.floor(i / (d) - (r * w));
            const g = Math.floor(i - (c * d) - (r * w * d));
            this.output_error.t[n][r][c][g] = dout.t[n][i];
        });
    }
    updateLayer() { }
}
exports.default = FlattenLayer;
