export default class Adam {
    constructor(layer) {
        this.name = "adam";
        this.t = 0;
        this.decay_rate_1 = 0.9;
        this.decay_rate_2 = 0.99;
        this.epsilon = Math.pow(10, -8);
        this.layer = layer;
        this.bias_first_moment = layer.bias.copy(false);
        this.bias_second_moment = layer.bias.copy(false);
        this.weight_first_moment = layer.weights.copy(false);
        this.weight_second_moment = layer.weights.copy(false);
    }
    optimizeWeights() {
        this.t += 1;
        this.weight_first_moment = this.weight_first_moment.mul(this.decay_rate_1).add(this.layer.errorWeights.mul(1 - this.decay_rate_1));
        this.weight_second_moment = this.weight_second_moment.mul(this.decay_rate_2).add(this.layer.errorWeights.pow(2).mul(1 - this.decay_rate_2));
        const first_moment_corrected = this.weight_first_moment.div(1 - (Math.pow(this.decay_rate_1, this.t)));
        const second_moment_corrected = this.weight_second_moment.div(1 - (Math.pow(this.decay_rate_2, this.t)));
        const w_update = first_moment_corrected.div(second_moment_corrected.sqrt().add(this.epsilon));
        this.layer.weights = this.layer.weights.sub(w_update.mul(this.layer.learning_rate));
    }
    optimizeBias() {
        this.bias_first_moment = this.bias_first_moment.mul(this.decay_rate_1).add(this.layer.errorBias.mul(1 - this.decay_rate_1));
        this.bias_second_moment = this.bias_second_moment.mul(this.decay_rate_2).add(this.layer.errorBias.pow(2).mul(1 - this.decay_rate_2));
        const first_moment_corrected = this.bias_first_moment.div(1 - (Math.pow(this.decay_rate_1, this.t)));
        const second_moment_corrected = this.bias_second_moment.div(1 - (Math.pow(this.decay_rate_2, this.t)));
        const b_update = first_moment_corrected.div(second_moment_corrected.sqrt().add(this.epsilon));
        this.layer.bias = this.layer.bias.sub(b_update.mul(this.layer.learning_rate));
    }
}
