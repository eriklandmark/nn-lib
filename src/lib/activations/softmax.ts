import Matrix from "../../matrix";
import IActivation from "./activations";
import {GPUFunction, KernelFunction, ThreadKernelVariable} from "gpu.js";

export default class Softmax implements IActivation{

    name: string = "softmax"

    normal(input: Matrix): Matrix {
        const exp = input.exp();
        return exp.div(exp.sum(1, true))
    }

    derivative(input: Matrix | number): Matrix | number {
        if (input instanceof Matrix) {
            const m = input.copy(false)
            m.iterate((i: number, j: number) => {
                if (i == j) {
                    m.set(i, j, input.get(i, j) * (1 - input.get(i, j)))
                } else {
                    m.set(i, j, -(input.get(i, j)*input.get(i,j)))
                }
            });
            return m
        } else {
            return input * (1 - input)
        }
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
        return function actv_der(a: number) {
            return a * (1 - a)
        }
    }
}