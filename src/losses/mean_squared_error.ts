import ILoss from "./losses";
import Matrix from "../matrix";
import {KernelFunction} from "gpu.js";

export default class MeanSquaredError implements ILoss {

    name: string = "mean_squared_error"

    normal(input: Matrix, labels: Matrix): Matrix {
        if (input.dim().r != labels.dim().r || input.dim().c != labels.dim().c)
            throw "Labels and output vector doesn't match size..";
        return input.sub(labels).pow(2)
    }

    derivative(input: Matrix, labels: Matrix): Matrix {
        if (input.dim().r != labels.dim().r || input.dim().c != labels.dim().c)
            throw "Labels and output vector doesn't match size..";
        return input.sub(labels)
    }

    normal_gpu(): KernelFunction {
        return function actv() {
            return
        }
    }

    derivative_gpu(): KernelFunction {
        return function loss(m, label) {
            //@ts-ignore
            return m - label
        }
    }
}