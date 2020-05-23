import ILoss from "./losses";
import {KernelFunction} from "gpu.js";
import Tensor from "../tensor";

export default class MeanSquaredError implements ILoss {

    name: string = "mean_squared_error"

    normal(input: Tensor, labels: Tensor): Tensor {
        if (!input.equalShape(labels))
            throw "Labels and output vector doesn't match size..";
        return input.sub(labels).pow(2)
    }

    derivative(input: Tensor, labels: Tensor): Tensor {
        if (!input.equalShape(labels))
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