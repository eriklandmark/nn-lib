import Matrix from "../matrix";
import IActivation from "./activations";
import { GPUFunction, KernelFunction, ThreadKernelVariable } from "gpu.js";
export default class Elu implements IActivation {
    name: string;
    a: number;
    normal(input: Matrix | number): Matrix | number;
    derivative(input: Matrix | number): Matrix | number;
    normal_gpu(): KernelFunction;
    derivative_gpu(): GPUFunction<ThreadKernelVariable[]>;
}
