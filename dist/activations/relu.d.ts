import IActivation from "./activations";
import { GPUFunction, KernelFunction, ThreadKernelVariable } from "gpu.js";
import Tensor from "../tensor";
export default class ReLu implements IActivation {
    name: string;
    normal(input: Tensor | number): Tensor | number;
    derivative(input: Tensor | number): Tensor | number;
    normal_gpu(): KernelFunction;
    derivative_gpu(): GPUFunction<ThreadKernelVariable[]>;
}
