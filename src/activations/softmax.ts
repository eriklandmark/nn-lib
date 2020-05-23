import IActivation from "./activations";
import {GPUFunction, KernelFunction, ThreadKernelVariable} from "gpu.js";
import Tensor from "../tensor";

export default class Softmax implements IActivation{

    name: string = "softmax"

    normal(input: Tensor): Tensor {
        const exp = input.exp();
        return exp.div(exp.sum(1, true), true)
    }

    derivative(input: Tensor | number): Tensor | number {
        if (input instanceof Tensor) {
            const m = input.copy(false)
            m.iterate((pos) => {
                if (pos[0] == pos[1]) {
                    m.set(pos, input.get(pos) * (1 - input.get(pos)))
                } else {
                    m.set(pos, -(input.get(pos)*input.get(pos)))
                }
            }, true);
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