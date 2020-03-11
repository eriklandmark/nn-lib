import Matrix from "../../matrix";
import {KernelFunction} from "gpu.js";

export default interface IActivation {
    name: string
    normal(input: Matrix | number): Matrix | number
    derivative(input: Matrix | number): Matrix | number
    normal_gpu(): KernelFunction
    derivative_gpu(): KernelFunction
}