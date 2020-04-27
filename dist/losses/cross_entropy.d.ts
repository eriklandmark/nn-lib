import ILoss from "./losses";
import Matrix from "../matrix";
import { KernelFunction } from "gpu.js";
export default class CrossEntropy implements ILoss {
    name: string;
    epsilon: number;
    normal(input: Matrix, labels: Matrix): Matrix;
    derivative(input: Matrix, labels: Matrix): Matrix;
    normal_gpu(): KernelFunction;
    derivative_gpu(): KernelFunction;
}
