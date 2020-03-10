import IActivation from "./activations";
import {KernelFunction} from "gpu.js";
import Matrix from "../../matrix";

export default class Sigmoid implements IActivation {

    name: string = "sigmoid"

    normal_gpu(): KernelFunction {
        return function actv(a) {
            return (1 / (1 + Math.exp(-a)))
        }
    }

    derivative_gpu(): KernelFunction {
        return function actv(a) {
            return (1 / (1 + Math.exp(-a)))
        }
    }

    normal(input: Matrix): Matrix {
        const m = input.copy()
        m.iterate((i: number, j: number) => {
            m.set(i, j, 1 / (1 + Math.exp(-input.get(i, j))))
        });
        return m
    }

    derivative(input: Matrix): Matrix {
        const m = input.copy()
        m.iterate((i: number, j: number) => {
            m.set(i, j, m.get(i, j) * (1 - input.get(i, j)))
        });
        return m
    }


}