import {IActivation} from "./activations";
import {GPUFunction, KernelFunction, ThreadKernelVariable} from "gpu.js";
import Tensor from "../tensor";

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

    normal(input: Tensor | number): Tensor | number {
        if (input instanceof Tensor) {
            const m = input.copy(false)
            m.iterate((pos) => {
                m.set(pos, Math.tanh(input.get(pos)))
            }, true);
            return m
        } else {
            return Math.tanh(input)
        }
    }

    derivative(input: Tensor | number): Tensor | number {
        if (input instanceof Tensor) {
            const m = input.copy(false)
            m.iterate((pos) => {
                m.set(pos, 1 - input.get(pos)**2)
            }, true);
            return m
        } else {
            return 1 - input**2
        }

    }
}