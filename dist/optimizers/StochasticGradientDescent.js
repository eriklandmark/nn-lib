"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const vector_1 = __importDefault(require("../vector"));
const matrix_1 = __importDefault(require("../matrix"));
class SochasticGradientDescent {
    constructor(layer) {
        this.name = "sgd";
        this.layer = layer;
    }
    optimizeWeights() {
        if (this.layer.weights instanceof matrix_1.default) {
            this.layer.weights = this.layer.weights.sub(this.layer.errorWeights.mul(this.layer.learning_rate));
        }
        else {
            for (let i = 0; i < this.layer.weights.length; i++) {
                this.layer.weights[i] = this.layer.weights[i].sub(this.layer.errorWeights[i].mul(this.layer.learning_rate));
            }
        }
    }
    optimizeBias() {
        if (this.layer.bias instanceof vector_1.default) {
            this.layer.bias = this.layer.bias.sub(this.layer.bias.mul(this.layer.learning_rate));
        }
        else if (this.layer.bias instanceof matrix_1.default) {
            this.layer.bias = this.layer.bias.sub(this.layer.bias.mul(this.layer.learning_rate));
        }
    }
}
exports.default = SochasticGradientDescent;
