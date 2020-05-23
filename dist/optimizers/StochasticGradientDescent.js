"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
class SochasticGradientDescent {
    constructor(layer) {
        this.name = "sgd";
        this.layer = layer;
    }
    optimizeWeights() {
        this.layer.weights = this.layer.weights.sub(this.layer.errorWeights.mul(this.layer.learning_rate));
    }
    optimizeBias() {
        this.layer.bias = this.layer.bias.sub(this.layer.errorBias.mul(this.layer.learning_rate));
    }
}
exports.default = SochasticGradientDescent;
