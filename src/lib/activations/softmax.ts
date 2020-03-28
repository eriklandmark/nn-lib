import Matrix from "../../matrix";
import IActivation from "./activations";
import {GPUFunction, KernelFunction, ThreadKernelVariable} from "gpu.js";

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
        return function actv(a: any[]) {
            let sum = 0;
            for (let i = 0; i < this.constants.softmax; i++) {
                sum += Math.exp(a[this.thread.y][i])
            }
            return Math.exp(a[this.thread.y][this.thread.x]) / sum
        }
    }

    derivative_gpu(): GPUFunction<ThreadKernelVariable[]> {
        return function actv_der(a) {
            return (1 / (1 + Math.exp(-a)))
        }
    }
}