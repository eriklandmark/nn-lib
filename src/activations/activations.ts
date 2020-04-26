import Matrix from "../matrix";
import {GPUFunction, KernelFunction, KernelVariable, ThreadKernelVariable} from "gpu.js";
import Sigmoid from "./sigmoid";
import ReLu from "./relu";
import Softmax from "./softmax";
import HyperbolicTangent from "./hyperbolic_tangent";

export interface IActivation {
    name: string
    normal(input: Matrix | number): Matrix | number
    derivative(input: Matrix | number): Matrix | number
    normal_gpu(): KernelFunction
    derivative_gpu(): GPUFunction<ThreadKernelVariable[]>
}

export default class Activation {

    static fromName(name: string): IActivation {
        switch (name) {
            case "sigmoid": return new Sigmoid()
            case "relu": return new ReLu()
            case "softmax": return new Softmax()
            case "tanh": return new HyperbolicTangent()

            default: return new Sigmoid()
        }
    }
}