import { KernelFunction } from "gpu.js";
import MeanSquaredError from "./mean_squared_error";
import Tensor from "../tensor";
export interface ILoss {
    name: string;
    normal(input: Tensor, labels: Tensor): Tensor;
    derivative(input: Tensor, labels: Tensor): Tensor;
    normal_gpu(): KernelFunction;
    derivative_gpu(): KernelFunction;
}
export default class Losses {
    static fromName(name: string): typeof MeanSquaredError;
}
