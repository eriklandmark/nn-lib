import {IOptimizer} from "./Optimizers";
import Layer from "../layers/layer";
import Tensor from "../tensor";

export default class SochasticGradientDescent implements IOptimizer {

    name: string = "sgd"
    layer: Layer

    constructor(layer: Layer) {
        this.layer = layer
    }

    optimizeWeights(): void {
        this.layer.weights = this.layer.weights.sub((<Tensor>this.layer.errorWeights).mul(this.layer.learning_rate))
    }

    optimizeBias(): void {
        this.layer.bias = this.layer.bias.sub(this.layer.errorBias.mul(this.layer.learning_rate))
    }
}