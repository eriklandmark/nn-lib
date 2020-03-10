import ILoss from "./losses";
import Matrix from "../../matrix";
import {KernelFunction} from "gpu.js";

export default class CrossEntropy implements ILoss {

    name: string = "cross_entropy"

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