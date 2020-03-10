import Matrix from "../../matrix";
import {KernelFunction} from "gpu.js";

export default interface IActivation {
    name: string
    normal(input: Matrix): Matrix
    derivative(input: Matrix): Matrix
    normal_gpu(): KernelFunction
    derivative_gpu(): KernelFunction
}