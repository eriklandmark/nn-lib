import Dataset, { Example } from "./dataset";
import Layer from "./layers/layer";
import Tensor from "./tensor";
import StochasticGradientDescent from "./optimizers/StochasticGradientDescent";
export interface SavedLayer {
    weights: any;
    bias: any;
    shape: number[];
    activation: string;
    prevLayerShape: number[];
    optimizer: string;
    layer_specific: {
        nr_filters?: number;
        filterSize?: number[];
        stride?: number[] | number;
        padding?: number;
        poolingFunc?: string;
        loss?: string;
        rate?: number;
        layerSize?: number;
        [propName: string]: any;
    };
    type?: string;
}
export interface ModelSettings {
    USE_GPU: boolean;
    BACKLOG: boolean;
    SAVE_CHECKPOINTS: boolean;
    MODEL_SAVE_PATH: string;
    VERBOSE_COMPACT: boolean;
    EVAL_PER_EPOCH: boolean;
    WORKER_MODE?: boolean;
    WORKER_CALLBACK?: Function;
}
export interface BacklogData {
    actual_duration: number;
    calculated_duration: number;
    train_start_time: number;
    info: {
        input_shape: number[];
        learning_rate: number;
        optimizer: string;
        loss: string;
        model_structure: any;
        total_neurons: number;
        batches_per_epoch: number;
        total_epochs: number;
        eval_model: boolean;
    };
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
            eval_loss: number;
            eval_accuracy: number;
        };
    };
}
export default class Model {
    layers: Layer[];
    gpuInstance: any;
    GPU: any;
    private isBuilt;
    backlog: BacklogData;
    settings: ModelSettings;
    constructor(layers: Layer[]);
    isGpuAvailable(): boolean;
    build(inputShape: number[], learning_rate: number, lossFunction: any, optimizer?: typeof StochasticGradientDescent, verbose?: boolean): void;
    summary(): void;
    train_on_batch(examples: Tensor, labels: Tensor): any;
    train(data: Example[] | Dataset, epochs: number, eval_ds?: Dataset | null, shuffle?: boolean): Promise<void>;
    saveBacklog(): void;
    eval(example: Example): any;
    predict(data: Tensor): Tensor;
    save(model_path?: string): void;
    load(path: string, verbose?: boolean): void;
}
