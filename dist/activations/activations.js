import Sigmoid from "./sigmoid";
import ReLu from "./relu";
import Softmax from "./softmax";
import HyperbolicTangent from "./hyperbolic_tangent";
import Elu from "./elu";
export default class Activation {
    static fromName(name) {
        switch (name) {
            case "sigmoid": return new Sigmoid();
            case "relu": return new ReLu();
            case "elu": return new Elu();
            case "softmax": return new Softmax();
            case "tanh": return new HyperbolicTangent();
            default: return new Sigmoid();
        }
    }
}
