import IActivation from "./activations";
import {GPUFunction, KernelFunction, ThreadKernelVariable} from "gpu.js";
import Tensor from "../tensor";

export default class Sigmoid implements IActivation {

    name: string = "sigmoid"

    normal_gpu(): KernelFunction {
        return function actv(a) {
            return (1 / (1 + Math.exp(-a[this.thread.y][this.thread.x])))
        }
    }

    derivative_gpu(): GPUFunction<ThreadKernelVariable[]> {
        return function actv_der(a: number) {
            return a * (1 - a)
        }
    }

    normal(input: Tensor | number): Tensor | number {
        if (input instanceof Tensor) {
            const m = input.copy(false)
            m.iterate((pos) => {
                m.set(pos, 1 / (1 + Math.exp(-input.get(pos))))
            }, true);
            return m
        } else {
            return 1 / (1 + Math.exp(-input))
        }
    }

    derivative(input: Tensor | number): Tensor | number {
        if (input instanceof Tensor) {
            const m = input.copy(false)
            m.iterate((pos) => {
                m.set(pos, input.get(pos) * (1 - input.get(pos)))
            }, true);
            return m
        } else {
            return input * (1 - input)
        }

    }



}