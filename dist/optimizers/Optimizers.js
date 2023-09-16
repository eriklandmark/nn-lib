import StochasticGradientDescent from "./StochasticGradientDescent";
import Adam from "./Adam";
export default class Optimizers {
    static fromName(name) {
        switch (name) {
            case "sgd": return StochasticGradientDescent;
            case "adam": return Adam;
        }
    }
}
