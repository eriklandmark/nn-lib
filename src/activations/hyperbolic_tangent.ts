import {IActivation} from "./activations";
import Matrix from "../matrix";
import {GPUFunction, KernelFunction, ThreadKernelVariable} from "gpu.js";

export default class HyperbolicTangent implements IActivation {
    name: string = "tanh";

    normal_gpu(): KernelFunction {
        return function actv(a: any[]) {
            return Math.tanh(a[this.thread.y][this.thread.x])
        }
    }

    derivative_gpu(): GPUFunction<ThreadKernelVariable[]> {
        return function actv_der(a: number) {
            return  1 - a**2
        }
    }

    normal(input: Matrix | number): Matrix | number {
        if (input instanceof Matrix) {
            const m = input.copy(false)
            m.iterate((i: number, j: number) => {
                m.set(i, j, Math.tanh(input.get(i, j)))
            });
            return m
        } else {
            return Math.tanh(input)
        }
    }

    derivative(input: Matrix | number): Matrix | number {
        if (input instanceof Matrix) {
            const m = input.copy(false)
            m.iterate((i: number, j: number) => {
                m.set(i, j, 1 - input.get(i, j)**2)
            });
            return m
        } else {
            return 1 - input**2
        }

    }
}