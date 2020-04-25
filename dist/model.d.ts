import Dataset, { Example } from "./dataset";
import Layer from "./lib/layers/layer";
import Matrix from "./matrix";
import Vector from "./vector";
import { GPU } from 'gpu.js';
import { ILoss } from "./lib/losses/losses";
import Tensor from "./tensor";
export interface SavedLayer {
    weights?: Float64Array[];
    bias?: Float64Array;
    shape?: number[];
    filters?: Float64Array[][][];
    nr_filters?: number;
    filterSize?: number[];
    activation?: string;
    loss?: string;
    rate?: number;
    prevLayerShape?: number[];
}
export default class Model {
    layers: Layer[];
    learning_rate: number;
    gpuInstance: GPU;
    USE_GPU: boolean;
    input_shape: number[];
    private isBuilt;
    constructor(layers: Layer[]);
    isGpuAvailable(): boolean;
    build(inputShape: number[], lossFunction: ILoss, verbose?: boolean): void;
    summary(): void;
    train_on_batch(examples: Matrix | Tensor[], labels: Matrix): number;
    train(data: Example[] | Dataset, epochs: number, learning_rate: number, shuffle?: boolean, verbose?: boolean): Promise<void>;
    predict(data: Vector | Matrix | Tensor): Matrix;
    save(path: string): void;
    load(path: string): void;
}
