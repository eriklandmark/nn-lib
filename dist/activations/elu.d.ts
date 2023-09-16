import IActivation from "./activations";
import { GPUFunction, KernelFunction, ThreadKernelVariable } from "gpu.js";
import Tensor from "../tensor";
export default class Elu implements IActivation {
    name: string;
    a: number;
    normal(input: Tensor | number): Tensor | number;
    derivative(input: Tensor | number): Tensor | number;
    normal_gpu(): KernelFunction;
    derivative_gpu(): GPUFunction<ThreadKernelVariable[]>;
}
