import {IOptimizer} from "./Optimizers";
import Matrix from "../matrix";
import Tensor from "../tensor";
import Layer from "../layers/layer";
import Vector from "../vector";

export default class Adam implements IOptimizer {

    name: string = "adam"
    layer: Layer
    weight_first_moment: Tensor[] | Matrix = []
    weight_second_moment: Tensor[] | Matrix = []
    bias_first_moment:  Matrix
    bias_second_moment: Matrix
    t: number = 0
    latest_t: number = 0
    decay_rate_1: number = 0.9
    decay_rate_2: number = 0.99
    epsilon: number = 10**-8

    constructor(layer: Layer) {
        this.layer = layer

        this.bias_first_moment = layer.bias.copy(false)
        this.bias_second_moment = layer.bias.copy(false)

        if(layer.weights instanceof Matrix) {
            this.weight_first_moment = layer.weights.copy(false)
            this.weight_second_moment = layer.weights.copy(false)
        } else {
            this.weight_first_moment = layer.weights.map((t) => t.copy(false))
            this.weight_second_moment = layer.weights.map((t) => t.copy(false))
        }
    }

    optimizeWeights(): void {
        this.t += 1
        if(this.weight_first_moment instanceof Matrix && this.weight_second_moment instanceof Matrix) {
            this.weight_first_moment = this.weight_first_moment.mul(this.decay_rate_1).add((<Matrix>this.layer.errorWeights).mul(1 - this.decay_rate_1))
            this.weight_second_moment = this.weight_second_moment.mul(this.decay_rate_2).add((<Matrix>this.layer.errorWeights).pow(2).mul(1 - this.decay_rate_2))
            const first_moment_corrected = this.weight_first_moment.div(1-(this.decay_rate_1 ** this.t))
            const second_moment_corrected = this.weight_second_moment.div(1-(this.decay_rate_2 ** this.t))
            const w_update = first_moment_corrected.div(second_moment_corrected.sqrt().add(this.epsilon))
            this.layer.weights = (<Matrix>this.layer.weights).sub(w_update.mul(this.layer.learning_rate))
        } else if(Array.isArray(this.weight_first_moment)) {
            for(let i = 0; i < this.weight_first_moment.length; i++) {
                this.weight_first_moment[i] = this.weight_first_moment[i].mul(this.decay_rate_1).add((<Tensor>this.layer.errorWeights[i]).mul(1 - this.decay_rate_1))
                this.weight_second_moment[i] = this.weight_second_moment[i].mul(this.decay_rate_2).add((<Tensor>this.layer.errorWeights[i]).pow(2).mul(1 - this.decay_rate_2))
                const first_moment_corrected = this.weight_first_moment[i].div(1-(this.decay_rate_1 ** this.t))
                const second_moment_corrected = this.weight_second_moment[i].div(1-(this.decay_rate_2 ** this.t))
                const w_update = first_moment_corrected.div(second_moment_corrected.sqrt().add(this.epsilon))
                this.layer.weights[i] = (<Tensor>this.layer.weights[i]).sub(w_update.mul(this.layer.learning_rate))
            }
        }
    }

    optimizeBias(): void {
        if(this.bias_first_moment instanceof Matrix && this.bias_second_moment instanceof Matrix) {
            this.bias_first_moment = this.bias_first_moment.mul(this.decay_rate_1).add((<Matrix>this.layer.errorBias).mul(1 - this.decay_rate_1))
            this.bias_second_moment = this.bias_second_moment.mul(this.decay_rate_2).add((<Matrix>this.layer.errorBias).pow(2).mul(1 - this.decay_rate_2))
            const first_moment_corrected = this.bias_first_moment.div(1-(this.decay_rate_1 ** this.t))
            const second_moment_corrected = this.bias_second_moment.div(1-(this.decay_rate_2 ** this.t))
            const b_update = first_moment_corrected.div(second_moment_corrected.sqrt().add(this.epsilon))
            this.layer.bias = (<Matrix>this.layer.bias).sub(b_update.mul(this.layer.learning_rate))
        }
    }
}