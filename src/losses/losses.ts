import {KernelFunction} from "gpu.js";
import CrossEntropy from "./cross_entropy";
import MeanSquaredError from "./mean_squared_error";
import Tensor from "../tensor";

export interface ILoss {
    name: string
    normal(input: Tensor, labels: Tensor): Tensor
    derivative(input: Tensor, labels: Tensor): Tensor
    normal_gpu(): KernelFunction
    derivative_gpu(): KernelFunction
}

export default class Losses {

    static fromName(name: string) {
        switch (name) {
            case "cross_entropy": return CrossEntropy
            case "mean_squared_error": return MeanSquaredError
        }
    }
}