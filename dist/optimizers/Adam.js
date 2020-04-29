"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const matrix_1 = __importDefault(require("../matrix"));
class Adam {
    constructor(layer) {
        this.name = "adam";
        this.weight_first_moment = [];
        this.weight_second_moment = [];
        this.t = 0;
        this.latest_t = 0;
        this.decay_rate_1 = 0.9;
        this.decay_rate_2 = 0.99;
        this.epsilon = Math.pow(10, -8);
        this.layer = layer;
        this.bias_first_moment = layer.bias.copy(false);
        this.bias_second_moment = layer.bias.copy(false);
        if (layer.weights instanceof matrix_1.default) {
            this.weight_first_moment = layer.weights.copy(false);
            this.weight_second_moment = layer.weights.copy(false);
        }
        else {
            this.weight_first_moment = layer.weights.map((t) => t.copy(false));
            this.weight_second_moment = layer.weights.map((t) => t.copy(false));
        }
    }
    optimizeWeights() {
        this.t += 1;
        if (this.weight_first_moment instanceof matrix_1.default && this.weight_second_moment instanceof matrix_1.default) {
            this.weight_first_moment = this.weight_first_moment.mul(this.decay_rate_1).add(this.layer.errorWeights.mul(1 - this.decay_rate_1));
            this.weight_second_moment = this.weight_second_moment.mul(this.decay_rate_2).add(this.layer.errorWeights.pow(2).mul(1 - this.decay_rate_2));
            const first_moment_corrected = this.weight_first_moment.div(1 - (Math.pow(this.decay_rate_1, this.t)));
            const second_moment_corrected = this.weight_second_moment.div(1 - (Math.pow(this.decay_rate_2, this.t)));
            const w_update = first_moment_corrected.div(second_moment_corrected.sqrt().add(this.epsilon));
            this.layer.weights = this.layer.weights.sub(w_update.mul(this.layer.learning_rate));
        }
        else if (Array.isArray(this.weight_first_moment)) {
            for (let i = 0; i < this.weight_first_moment.length; i++) {
                this.weight_first_moment[i] = this.weight_first_moment[i].mul(this.decay_rate_1).add(this.layer.errorWeights[i].mul(1 - this.decay_rate_1));
                this.weight_second_moment[i] = this.weight_second_moment[i].mul(this.decay_rate_2).add(this.layer.errorWeights[i].pow(2).mul(1 - this.decay_rate_2));
                const first_moment_corrected = this.weight_first_moment[i].div(1 - (Math.pow(this.decay_rate_1, this.t)));
                const second_moment_corrected = this.weight_second_moment[i].div(1 - (Math.pow(this.decay_rate_2, this.t)));
                const w_update = first_moment_corrected.div(second_moment_corrected.sqrt().add(this.epsilon));
                this.layer.weights[i] = this.layer.weights[i].sub(w_update.mul(this.layer.learning_rate));
            }
        }
    }
    optimizeBias() {
        if (this.bias_first_moment instanceof matrix_1.default && this.bias_second_moment instanceof matrix_1.default) {
            this.bias_first_moment = this.bias_first_moment.mul(this.decay_rate_1).add(this.layer.errorBias.mul(1 - this.decay_rate_1));
            this.bias_second_moment = this.bias_second_moment.mul(this.decay_rate_2).add(this.layer.errorBias.pow(2).mul(1 - this.decay_rate_2));
            const first_moment_corrected = this.bias_first_moment.div(1 - (Math.pow(this.decay_rate_1, this.t)));
            const second_moment_corrected = this.bias_second_moment.div(1 - (Math.pow(this.decay_rate_2, this.t)));
            const b_update = first_moment_corrected.div(second_moment_corrected.sqrt().add(this.epsilon));
            this.layer.bias = this.layer.bias.sub(b_update.mul(this.layer.learning_rate));
        }
    }
}
exports.default = Adam;
