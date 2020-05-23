import IActivation from "./activations";
import {GPUFunction, KernelFunction, ThreadKernelVariable} from "gpu.js";
import Tensor from "../tensor";

export default class ReLu implements IActivation {

    name: string = "relu"

    normal(input: Tensor | number): Tensor | number {
        if (input instanceof Tensor) {
            const m = input.copy(false)
            m.iterate((pos) => {
                m.set(pos, Math.max(input.get(pos), 0))
            }, true);
            return m
        } else {
            return Math.max(input, 0)
        }
    }

    derivative(input: Tensor | number): Tensor | number {
        if (input instanceof Tensor) {
            const m = input.copy(false)
            m.iterate((pos) => {
                m.set(pos, input.get(pos) > 0 ? 1 : 0)
            }, true);
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