import StochasticGradientDescent from "./StochasticGradientDescent";
export interface IOptimizer {
    name: string;
    optimizeWeights(): void;
    optimizeBias(): void;
}
export default class Optimizers {
    static fromName(name: string): typeof StochasticGradientDescent;
}
