import ILoss from "./losses";
import Matrix from "../../matrix";
import {KernelFunction} from "gpu.js";

export default class MeanSquaredError implements ILoss {
    name: string;
    normal(input: Matrix, labels: Matrix): Matrix;
    derivative(input: Matrix, labels: Matrix): Matrix;
    normal_gpu(): KernelFunction;
    derivative_gpu(): KernelFunction;
}
