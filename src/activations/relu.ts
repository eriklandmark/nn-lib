import Matrix from "../matrix";
import IActivation from "./activations";
import {GPUFunction, KernelFunction, ThreadKernelVariable} from "gpu.js";

export default class ReLu implements IActivation {

    name: string = "relu"

    normal(input: Matrix | number): Matrix | number {
        if (input instanceof Matrix) {
            const m = input.copy(false)
            m.iterate((i: number, j: number) => {
                m.set(i, j, Math.max(input.get(i,j), 0))
            });
            return m
        } else {
            return Math.max(input, 0)
        }
    }

    derivative(input: Matrix | number): Matrix | number {
        if (input instanceof Matrix) {
            const m = input.copy(false)
            m.iterate((i: number, j: number) => {
                m.set(i, j, input.get(i,j) > 0 ? 1 : 0)
            });
            return m
        } else {
            return input > 0 ? 1 : 0
        }
    }

    normal_gpu(): KernelFunction {
        return function actv(a) {
            return Math.max(a[this.thread.y][this.thread.x], 0)
        }
    }

    derivative_gpu(): GPUFunction<ThreadKernelVariable[]> {
        return function actv(a: any) {
            return a > 0 ? 1 : 0
        }
    }
}