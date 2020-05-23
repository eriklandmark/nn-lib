import StochasticGradientDescent from "./StochasticGradientDescent";
import Adam from "./Adam";

export interface IOptimizer {
    name: string
    optimizeWeights(): void
    optimizeBias(): void
}

export default class Optimizers {

    public static fromName(name: string) {
        switch (name) {
            case "sgd": return StochasticGradientDescent
            case "adam": return Adam
        }
    }
}