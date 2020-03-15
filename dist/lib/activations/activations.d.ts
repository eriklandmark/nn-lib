import Matrix from "../../matrix";
import {KernelFunction} from "gpu.js";

export interface IActivation {
    name: string;
    normal(input: Matrix | number): Matrix | number;
    derivative(input: Matrix | number): Matrix | number;
    normal_gpu(): KernelFunction;
    derivative_gpu(): KernelFunction;
}
export default class Activation {
    static fromName(name: string): IActivation;
}
