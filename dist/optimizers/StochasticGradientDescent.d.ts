import { IOptimizer } from "./Optimizers";
import Layer from "../layers/layer";
export default class SochasticGradientDescent implements IOptimizer {
    name: string;
    layer: Layer;
    constructor(layer: Layer);
    optimizeWeights(): void;
    optimizeBias(): void;
}
