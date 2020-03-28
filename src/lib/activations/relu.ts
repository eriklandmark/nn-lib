import Matrix from "../../matrix";
import IActivation from "./activations";
import {GPUFunction, KernelFunction, ThreadKernelVariable} from "gpu.js";

export default class ReLu implements IActivation {

    name: string = "relu"

    normal(input: Matrix): Matrix {
        const m = input.copy()
        m.iterate((i: number, j: number) => {
            m.set(i, j, Math.max(m.get(i,j), 0))
        });
        return m
    }

    derivative(input: Matrix): Matrix {
        const m = input.copy()
        m.iterate((i: number, j: number) => {
            m.set(i, j, m.get(i,j) > 0 ? 1 : 0)
        });
        return m
    }

    normal_gpu(): KernelFunction {
        return function actv(a) {}
    }

    derivative_gpu(): GPUFunction<ThreadKernelVariable[]> {
        return function actv(a: any) {
            return a
        }
    }
}