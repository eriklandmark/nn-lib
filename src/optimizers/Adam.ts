import {IOptimizer} from "./Optimizers";
import Tensor from "../tensor";
import Layer from "../layers/layer";

export default class Adam implements IOptimizer {

    name: string = "adam"
    layer: Layer
    weight_first_moment: Tensor
    weight_second_moment: Tensor
    bias_first_moment:  Tensor
    bias_second_moment: Tensor
    t: number = 0
    decay_rate_1: number = 0.9
    decay_rate_2: number = 0.99
    epsilon: number = 10**-8

    constructor(layer: Layer) {
        this.layer = layer

        this.bias_first_moment = layer.bias.copy(false)
        this.bias_second_moment = layer.bias.copy(false)
        this.weight_first_moment = layer.weights.copy(false)
        this.weight_second_moment = layer.weights.copy(false)
    }

    optimizeWeights(): void {
        this.t += 1
        this.weight_first_moment = this.weight_first_moment.mul(this.decay_rate_1).add((<Tensor>this.layer.errorWeights).mul(1 - this.decay_rate_1))
        this.weight_second_moment = this.weight_second_moment.mul(this.decay_rate_2).add((<Tensor>this.layer.errorWeights).pow(2).mul(1 - this.decay_rate_2))
        const first_moment_corrected = this.weight_first_moment.div(1-(this.decay_rate_1 ** this.t))
        const second_moment_corrected = this.weight_second_moment.div(1-(this.decay_rate_2 ** this.t))
        const w_update = first_moment_corrected.div(second_moment_corrected.sqrt().add(this.epsilon))
        this.layer.weights = (<Tensor>this.layer.weights).sub(w_update.mul(this.layer.learning_rate))
    }

    optimizeBias(): void {
        this.bias_first_moment = this.bias_first_moment.mul(this.decay_rate_1).add((<Tensor>this.layer.errorBias).mul(1 - this.decay_rate_1))
        this.bias_second_moment = this.bias_second_moment.mul(this.decay_rate_2).add((<Tensor>this.layer.errorBias).pow(2).mul(1 - this.decay_rate_2))
        const first_moment_corrected = this.bias_first_moment.div(1-(this.decay_rate_1 ** this.t))
        const second_moment_corrected = this.bias_second_moment.div(1-(this.decay_rate_2 ** this.t))
        const b_update = first_moment_corrected.div(second_moment_corrected.sqrt().add(this.epsilon))
        this.layer.bias = (<Tensor>this.layer.bias).sub(b_update.mul(this.layer.learning_rate))
    }
}