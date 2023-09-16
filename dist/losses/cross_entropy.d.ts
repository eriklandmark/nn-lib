import ILoss from "./losses";
import { KernelFunction } from "gpu.js";
import Tensor from "../tensor";
export default class CrossEntropy implements ILoss {
    name: string;
    epsilon: number;
    normal(input: Tensor, labels: Tensor): Tensor;
    derivative(input: Tensor, labels: Tensor): Tensor;
    normal_gpu(): KernelFunction;
    derivative_gpu(): KernelFunction;
}
