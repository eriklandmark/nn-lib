import { IOptimizer } from "./Optimizers";
import Tensor from "../tensor";
import Layer from "../layers/layer";
export default class Adam implements IOptimizer {
    name: string;
    layer: Layer;
    weight_first_moment: Tensor;
    weight_second_moment: Tensor;
    bias_first_moment: Tensor;
    bias_second_moment: Tensor;
    t: number;
    decay_rate_1: number;
    decay_rate_2: number;
    epsilon: number;
    constructor(layer: Layer);
    optimizeWeights(): void;
    optimizeBias(): void;
}
