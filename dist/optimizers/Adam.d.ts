import { IOptimizer } from "./Optimizers";
import Matrix from "../matrix";
import Tensor from "../tensor";
import Layer from "../layers/layer";
export default class Adam implements IOptimizer {
    name: string;
    layer: Layer;
    weight_first_moment: Tensor[] | Matrix;
    weight_second_moment: Tensor[] | Matrix;
    bias_first_moment: Matrix;
    bias_second_moment: Matrix;
    t: number;
    latest_t: number;
    decay_rate_1: number;
    decay_rate_2: number;
    epsilon: number;
    constructor(layer: Layer);
    optimizeWeights(): void;
    optimizeBias(): void;
}
