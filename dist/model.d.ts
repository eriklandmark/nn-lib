import Dataset, {Example} from "./dataset";
import Layer from "./lib/layers/layer";
import Matrix from "./matrix";
import Vector from "./vector";
import {GPU} from 'gpu.js';
import ILoss from "./lib/losses/losses";

export default class Model {
    layers: Layer[];
    learning_rate: number;
    gpuInstance: GPU;
    USE_GPU: boolean;
    private isBuilt;
    constructor(layers: Layer[]);
    isGpuAvailable(): boolean;
    build(inputShape: number[], lossFunction: ILoss, verbose?: boolean): void;
    train_on_batch(examples: Matrix, labels: Matrix): number;
    train(data: Example[] | Dataset, epochs: number, learning_rate: number, shuffle?: boolean, verbose?: boolean): Promise<void>;
    predict(data: Vector | Matrix): Matrix;
    save(path: string): void;
    load(path: string): void;
}
