"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const sigmoid_1 = __importDefault(require("./sigmoid"));
const relu_1 = __importDefault(require("./relu"));
const softmax_1 = __importDefault(require("./softmax"));
const hyperbolic_tangent_1 = __importDefault(require("./hyperbolic_tangent"));
const elu_1 = __importDefault(require("./elu"));
class Activation {
    static fromName(name) {
        switch (name) {
            case "sigmoid": return new sigmoid_1.default();
            case "relu": return new relu_1.default();
            case "elu": return new elu_1.default();
            case "softmax": return new softmax_1.default();
            case "tanh": return new hyperbolic_tangent_1.default();
            default: return new sigmoid_1.default();
        }
    }
}
exports.default = Activation;
