import {IOptimizer} from "./Optimizers";
import Layer from "../layers/layer";
import Vector from "../vector";
import Matrix from "../matrix";

export default class SochasticGradientDescent implements IOptimizer {

    name: string = "sgd"
    layer: Layer

    constructor(layer: Layer) {
        this.layer = layer
    }

    optimizeWeights(): void {
        if (this.layer.weights instanceof Matrix) {
            this.layer.weights = this.layer.weights.sub((<Matrix>this.layer.errorWeights).mul(this.layer.learning_rate))
        } else {
            for(let i = 0; i < this.layer.weights.length; i++) {
                this.layer.weights[i] = this.layer.weights[i].sub(this.layer.errorWeights[i].mul(this.layer.learning_rate))
            }
        }
    }

    optimizeBias(): void {
        if (this.layer.bias instanceof Vector) {
            this.layer.bias = this.layer.bias.sub(this.layer.bias.mul(this.layer.learning_rate))
        } else if (this.layer.bias instanceof Matrix) {
            this.layer.bias = this.layer.bias.sub(this.layer.bias.mul(this.layer.learning_rate))
        }
    }
}