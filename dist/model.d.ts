import Dataset, { Example } from "./dataset";
import Layer from "./layers/layer";
import Matrix from "./matrix";
import Vector from "./vector";
import { GPU } from 'gpu.js';
import { ILoss } from "./losses/losses";
import Tensor from "./tensor";
export interface SavedLayer {
    weights?: Float32Array[];
    bias?: Float32Array | Float32Array[];
    shape?: number[];
    filters?: Float32Array[][][];
    nr_filters?: number;
    filterSize?: number[];
    activation?: string;
    loss?: string;
    rate?: number;
    prevLayerShape?: number[];
    stride?: number[] | number;
    padding?: number;
    poolingFunc?: string;
}
interface ModelSettings {
    USE_GPU: boolean;
    BACKLOG: boolean;
    SAVE_CHECKPOINTS: boolean;
    MODEL_SAVE_PATH: string;
}
export interface BacklogData {
    actual_duration: number;
    calculated_duration: number;
    epochs: {
        [propName: string]: {
            total_loss: number;
            total_accuracy: number;
            batches: {
                accuracy: number;
                loss: number;
                time: number;
            }[];
            calculated_duration: number;
            actual_duration: number;
        };
    };
}
export default class Model {
    layers: Layer[];
    gpuInstance: GPU;
    private isBuilt;
    backlog: BacklogData;
    settings: ModelSettings;
    model_data: {
        input_shape: number[];
        learning_rate: number;
        last_epoch: number;
    };
    constructor(layers: Layer[]);
    isGpuAvailable(): boolean;
    build(inputShape: number[], lossFunction: ILoss, verbose?: boolean): void;
    summary(): void;
    train_on_batch(examples: Matrix | Tensor[], labels: Matrix): any;
    train(data: Example[] | Dataset, epochs: number, learning_rate: number, shuffle?: boolean, verbose?: boolean): Promise<void>;
    saveBacklog(): void;
    predict(data: Vector | Matrix | Tensor): Matrix;
    save(model_path?: string): void;
    load(path: string, verbose?: boolean): void;
}
export {};
