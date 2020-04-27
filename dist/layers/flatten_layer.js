"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const layer_1 = __importDefault(require("./layer"));
const tensor_1 = __importDefault(require("../tensor"));
const matrix_1 = __importDefault(require("../matrix"));
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
        const matrix = new matrix_1.default(input.activation.map((tensor) => tensor.vectorize(true)));
        this.activation = matrix.transpose();
    }
    backPropagation(prev_layer, next_layer) {
        let error;
        if (prev_layer.output_error instanceof matrix_1.default) {
            error = prev_layer.output_error;
        }
        else {
            error = new matrix_1.default(prev_layer.output_error.toArray());
        }
        const dout = error.mm(prev_layer.weights.transpose());
        let t = new Array(error.dim().r);
        for (let i = 0; i < t.length; i++) {
            t[i] = new tensor_1.default();
            t[i].createEmptyArray(this.prevShape[0], this.prevShape[1], this.prevShape[2]);
        }
        let [h, w, d] = this.prevShape;
        dout.iterate((n, i) => {
            const r = Math.floor(i / (w * d));
            const c = Math.floor(i / (d) - (r * w));
            const g = Math.floor(i - (c * d) - (r * w * d));
            t[n].set(r, c, g, dout.get(n, i));
        });
        this.output_error = t;
    }
    toSavedModel() {
        return {
            shape: this.prevShape
        };
    }
    fromSavedModel(data) {
        this.buildLayer(data.shape);
    }
}
exports.default = FlattenLayer;
