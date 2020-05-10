import Matrix from "../matrix";
import IActivation from "./activations";
import {GPUFunction, KernelFunction, ThreadKernelVariable} from "gpu.js";

export default class Elu implements IActivation {

    name: string = "elu"
    a = 1

    normal(input: Matrix | number): Matrix | number {
        if (input instanceof Matrix) {
            const m = input.copy(false)
            m.iterate((i: number, j: number) => {
                m.set(i, j, input.get(i,j) > 0? input.get(i,j): this.a*(Math.exp(input.get(i,j)) - 1))
            });
            return m
        } else {
            return input > 0? input: this.a*(Math.exp(input) - 1)
        }
    }

    derivative(input: Matrix | number): Matrix | number {
        if (input instanceof Matrix) {
            const m = input.copy(false)
            m.iterate((i: number, j: number) => {
                m.set(i, j, input.get(i,j) > 0 ? 1 : input.get(i,j) + this.a)
            });
            return m
        } else {
            return input > 0 ? 1 : input + this.a
        }
    }

    normal_gpu(): KernelFunction {
        return function actv(a: number) {
            return a > 0? a : Math.exp(a) - 1
        }
    }

    derivative_gpu(): GPUFunction<ThreadKernelVariable[]> {
        return function actv(a: any) {
            return a > 0 ? 1 : a + 1
        }
    }
}