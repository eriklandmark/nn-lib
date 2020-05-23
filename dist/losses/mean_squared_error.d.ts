import ILoss from "./losses";
import { KernelFunction } from "gpu.js";
import Tensor from "../tensor";
export default class MeanSquaredError implements ILoss {
    name: string;
    normal(input: Tensor, labels: Tensor): Tensor;
    derivative(input: Tensor, labels: Tensor): Tensor;
    normal_gpu(): KernelFunction;
    derivative_gpu(): KernelFunction;
}
