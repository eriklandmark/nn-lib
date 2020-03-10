import Matrix from "../../matrix";
import IActivation from "./activations";
import {KernelFunction} from "gpu.js";

export default class Softmax implements IActivation{

    name: string = "softmax"

    normal(input: Matrix): Matrix {
        const exp = input.exp();
        return exp.div(exp.sum(1, true))
    }

    derivative(input: Matrix): Matrix {
        return input
    }

    normal_gpu(): KernelFunction {
        return function actv() {}
    }

    derivative_gpu(): KernelFunction {
        return function actv() {}
    }
}