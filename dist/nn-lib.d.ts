/// <reference types="node" />
declare module "matrix" {
    import Vector from "vector";
    import { KernelFunction } from "gpu.js";
    export default class Matrix {
        matrix: Array<Float32Array>;
        get: Function;
        set: Function;
        count: Function;
        constructor(defaultValue?: Array<Array<number>> | Array<Float32Array> | Array<Vector>);
        createEmptyArray(rows: number, columns: number): void;
        dim(): {
            r: number;
            c: number;
        };
        toString: (max_rows?: number, precision?: number) => string;
        private numberToString;
        static fromJsonObject(obj: any[]): Matrix;
        toNumberArray(): number[][];
        copy(full?: boolean): Matrix;
        fill(scalar: number): Matrix;
        iterate(func: Function): void;
        where(scalar: number): number[];
        populateRandom(): void;
        empty(): boolean;
        isNaN(): boolean;
        repeat(axis?: number, times?: number): Matrix;
        static addGpu(): KernelFunction;
        static subGpu(): KernelFunction;
        static multiplyGpu(): KernelFunction;
        static mmGpu(): KernelFunction;
        mm(b: Matrix | Vector, gpu?: boolean): Matrix | Vector;
        mmAsync(b: Matrix | Vector): Promise<Matrix | Vector>;
        add(b: number | Matrix): Matrix;
        sub(b: number | Matrix): Matrix;
        mul(b: number | Matrix): Matrix;
        pow(scalar: number): Matrix;
        sqrt(): Matrix;
        inv_el(eps?: number): Matrix;
        exp(): Matrix;
        log(): Matrix;
        sum(axis?: number, keepDims?: boolean): number | Matrix;
        div(scalar: number | Matrix): Matrix;
        transpose(): Matrix;
        argmax(i?: number, row?: boolean): number;
        inv(): Matrix;
        rowVectors(): Vector[];
        mean(axis?: number, keep_dims?: boolean): number | Matrix;
    }
}
declare module "tensor" {
    import Vector from "vector";
    import Matrix from "matrix";
    export default class Tensor {
        tensor: Float32Array[][];
        get: Function;
        set: Function;
        count: Function;
        dim(): {
            r: number;
            c: number;
            d: number;
        };
        shape(): number[];
        constructor(v?: number[][][] | Float32Array[][]);
        createEmptyArray(rows: number, columns: number, depth: number): void;
        static fromJsonObject(obj: any[][]): Tensor;
        toNumberArray(): number[][];
        iterate(func: Function, channel_first?: boolean): void;
        toString: (max_rows?: number) => string;
        copy(full?: boolean): Tensor;
        populateRandom(): void;
        empty(): boolean;
        vectorize(channel_first?: boolean): Vector;
        div(val: number | Tensor): Tensor;
        mul(val: number | Tensor): Tensor;
        sub(val: number | Tensor): Tensor;
        add(val: number | Tensor): Tensor;
        padding(padding_height: number, padding_width: number): Tensor;
        im2patches(patch_height: number, patch_width: number, filter_height: number, filter_width: number): Matrix;
        rotate180(): Tensor;
    }
}
declare module "vector" {
    import Tensor from "tensor";
    export default class Vector {
        vector: Float32Array;
        size: Function;
        get: Function;
        set: Function;
        constructor(defaultValue?: Float32Array | number[] | number);
        static fromJsonObj(obj: any): Vector;
        static fromBuffer(buff: Buffer): Vector;
        static toCategorical(index: number, size: number): Vector;
        toString: (vertical?: boolean) => string;
        toNumberArray(): any;
        populateRandom(): void;
        iterate(func: Function): void;
        add(b: number | Vector): Vector;
        sub(b: number | Vector): Vector;
        mul(input: number | Vector): Vector;
        div(scalar: number): Vector;
        pow(scalar: number): Vector;
        exp(): Vector;
        sum(): number;
        mean(): number;
        argmax(): number;
        reshape(shape: number[]): Tensor;
        normalize(): Vector;
    }
}
declare module "helpers/array_helper" {
    export default class ArrayHelper {
        static shuffle(array: any[]): any[];
        static flatten(array: any[]): any[];
        static delete_doublets(array: any[]): any[];
    }
}
declare module "dataset" {
    import Vector from "vector";
    import Tensor from "tensor";
    import Matrix from "matrix";
    export interface Example {
        data: Vector | Matrix | Tensor;
        label: Vector;
    }
    export default class Dataset {
        private data;
        BATCH_SIZE: number;
        IS_GENERATOR: boolean;
        TOTAL_EXAMPLES: number;
        DATA_STRUCTURE: any;
        GENERATOR: Function;
        size(): number;
        setGenerator(gen: Function): void;
        static read_image(path: string): Promise<Tensor>;
        vectorize_image(image: Tensor): Vector;
        loadMnistTrain(folderPath: string, maxExamples?: number, vectorize?: boolean): void;
        loadMnistTest(folderPath: string, maxExamples?: number, vectorize?: boolean): void;
        shuffle(): void;
        private loadMnist;
        loadTestData(path: string, maxExamples?: number): void;
        getBatch(batch: number): Example[];
    }
}
declare module "main" {
    export const NAME = "nn-lib";
}
declare module "activations/sigmoid" {
    import IActivation from "activations/activations";
    import { GPUFunction, KernelFunction, ThreadKernelVariable } from "gpu.js";
    import Matrix from "matrix";
    export default class Sigmoid implements IActivation {
        name: string;
        normal_gpu(): KernelFunction;
        derivative_gpu(): GPUFunction<ThreadKernelVariable[]>;
        normal(input: Matrix | number): Matrix | number;
        derivative(input: Matrix | number): Matrix | number;
    }
}
declare module "activations/relu" {
    import Matrix from "matrix";
    import IActivation from "activations/activations";
    import { GPUFunction, KernelFunction, ThreadKernelVariable } from "gpu.js";
    export default class ReLu implements IActivation {
        name: string;
        normal(input: Matrix | number): Matrix | number;
        derivative(input: Matrix | number): Matrix | number;
        normal_gpu(): KernelFunction;
        derivative_gpu(): GPUFunction<ThreadKernelVariable[]>;
    }
}
declare module "activations/softmax" {
    import Matrix from "matrix";
    import IActivation from "activations/activations";
    import { GPUFunction, KernelFunction, ThreadKernelVariable } from "gpu.js";
    export default class Softmax implements IActivation {
        name: string;
        normal(input: Matrix): Matrix;
        derivative(input: Matrix | number): Matrix | number;
        normal_gpu(): KernelFunction;
        derivative_gpu(): GPUFunction<ThreadKernelVariable[]>;
    }
}
declare module "activations/hyperbolic_tangent" {
    import { IActivation } from "activations/activations";
    import Matrix from "matrix";
    import { GPUFunction, KernelFunction, ThreadKernelVariable } from "gpu.js";
    export default class HyperbolicTangent implements IActivation {
        name: string;
        normal_gpu(): KernelFunction;
        derivative_gpu(): GPUFunction<ThreadKernelVariable[]>;
        normal(input: Matrix | number): Matrix | number;
        derivative(input: Matrix | number): Matrix | number;
    }
}
declare module "activations/activations" {
    import Matrix from "matrix";
    import { GPUFunction, KernelFunction, ThreadKernelVariable } from "gpu.js";
    export interface IActivation {
        name: string;
        normal(input: Matrix | number): Matrix | number;
        derivative(input: Matrix | number): Matrix | number;
        normal_gpu(): KernelFunction;
        derivative_gpu(): GPUFunction<ThreadKernelVariable[]>;
    }
    export default class Activation {
        static fromName(name: string): IActivation;
    }
}
declare module "layers/layer" {
    import Matrix from "matrix";
    import Vector from "vector";
    import { GPU } from "gpu.js";
    import { IActivation } from "activations/activations";
    import Tensor from "tensor";
    import { SavedLayer } from "model";
    export default class Layer {
        weights: Matrix;
        bias: Vector | Matrix;
        errorWeights: Matrix;
        errorBias: Matrix | Vector;
        output_error: any;
        activation: Matrix | Tensor[];
        activationFunction: IActivation;
        useGpu: boolean;
        gpuInstance: GPU;
        shape: number[];
        prevLayerShape: number[];
        type: string;
        hasGPUSupport: boolean;
        isFirstLayer: boolean;
        ff_kernel: any;
        act_kernel: any;
        bp_error_kernel: any;
        bp_error_weight_kernel: any;
        setGpuInstance(gpuIns: GPU): void;
        getLayerInfo(): {
            type: string;
            shape: number[];
            activation: string;
        };
        buildLayer(prevLayerShape: number[]): void;
        feedForward(input: Layer | Matrix | Tensor[], isInTraining: boolean): void;
        buildFFKernels(batch_size: number): void;
        buildBPKernels(size: number): void;
        backPropagation(prev_layer: Layer, next_layer: Layer | Matrix | Tensor[]): void;
        calculate_errors(error: any, next_layer: Layer | Matrix): void;
        updateWeights(l_rate: number): void;
        toSavedModel(): SavedLayer;
        fromSavedModel(data: SavedLayer): void;
    }
}
declare module "losses/cross_entropy" {
    import ILoss from "losses/losses";
    import Matrix from "matrix";
    import { KernelFunction } from "gpu.js";
    export default class CrossEntropy implements ILoss {
        name: string;
        epsilon: number;
        normal(input: Matrix, labels: Matrix): Matrix;
        derivative(input: Matrix, labels: Matrix): Matrix;
        normal_gpu(): KernelFunction;
        derivative_gpu(): KernelFunction;
    }
}
declare module "losses/mean_squared_error" {
    import ILoss from "losses/losses";
    import Matrix from "matrix";
    import { KernelFunction } from "gpu.js";
    export default class MeanSquaredError implements ILoss {
        name: string;
        normal(input: Matrix, labels: Matrix): Matrix;
        derivative(input: Matrix, labels: Matrix): Matrix;
        normal_gpu(): KernelFunction;
        derivative_gpu(): KernelFunction;
    }
}
declare module "losses/losses" {
    import Vector from "vector";
    import Matrix from "matrix";
    import { KernelFunction } from "gpu.js";
    import CrossEntropy from "losses/cross_entropy";
    import MeanSquaredError from "losses/mean_squared_error";
    export interface ILoss {
        name: string;
        normal(input: Matrix, labels: Matrix): Matrix;
        derivative(input: Matrix, labels: Matrix): Matrix;
        normal_gpu(): KernelFunction;
        derivative_gpu(): KernelFunction;
    }
    export default class Losses {
        static fromName(name: string): CrossEntropy | MeanSquaredError;
        static CrossEntropy(v: Vector, labels: Vector): Vector;
        static CrossEntropy_derivative(v: Vector, labels: Vector): Vector;
    }
}
declare module "layers/conv_layer" {
    import Layer from "layers/layer";
    import Tensor from "tensor";
    import { IActivation } from "activations/activations";
    import { SavedLayer } from "model";
    export default class ConvolutionLayer extends Layer {
        filterSize: number[];
        filters: Tensor[];
        padding: number;
        stride: number;
        nr_filters: number;
        errorFilters: Tensor[];
        errorInput: Tensor[];
        channel_first: boolean;
        ff_kernel: any;
        act_kernel: any;
        bp_error_kernel: any;
        bp_error_weight_kernel: any;
        useMM: boolean;
        constructor(nr_filters: number, filterSize: number[], ch_first: boolean, activation: IActivation);
        buildLayer(prevLayerShape: number[]): void;
        feedForward(input: Layer | Tensor[], isInTraining: boolean): void;
        buildFFKernels(batch_size: number): void;
        backPropagation(prev_layer: Layer, next_layer: Layer | Tensor[]): void;
        convolve(image: Tensor, filters: Tensor[], channel_first?: boolean): Tensor | Tensor[];
        updateWeights(l_rate: number): void;
        toSavedModel(): SavedLayer;
        fromSavedModel(data: SavedLayer): void;
    }
}
declare module "layers/dense_layer" {
    import Layer from "layers/layer";
    import Matrix from "matrix";
    import { IActivation } from "activations/activations";
    import { SavedLayer } from "model";
    export default class DenseLayer extends Layer {
        layerSize: number;
        ff_kernel: any;
        act_kernel: any;
        bp_error_kernel: any;
        bp_error_weight_kernel: any;
        constructor(layerSize?: number, activation?: IActivation);
        buildLayer(prevLayerShape: number[]): void;
        buildFFKernels(batch_size: number): void;
        buildBPKernels(length: number): void;
        feedForward(input: Layer | Matrix, isInTraining: boolean): any;
        calculate_errors(error: any, input: Matrix): void;
        backPropagation(prev_layer: Layer, next_layer: Layer | Matrix): void;
        updateWeights(l_rate: number): void;
        toSavedModel(): SavedLayer;
        fromSavedModel(data: SavedLayer): void;
    }
}
declare module "layers/dropout_layer" {
    import Layer from "layers/layer";
    import Matrix from "matrix";
    import { SavedLayer } from "model";
    export default class DropoutLayer extends Layer {
        rate: number;
        type: string;
        constructor(rate?: number);
        buildLayer(prevLayerShape: number[]): void;
        feedForward(input: Layer | Matrix, isInTraining: boolean): void;
        backPropagation(prev_layer: Layer, next_layer: Layer | Matrix): void;
        updateWeights(l_rate: number): void;
        toSavedModel(): SavedLayer;
        fromSavedModel(data: SavedLayer): void;
    }
}
declare module "layers/flatten_layer" {
    import Layer from "layers/layer";
    import Matrix from "matrix";
    import { SavedLayer } from "model";
    export default class FlattenLayer extends Layer {
        type: string;
        prevShape: number[];
        buildLayer(prevLayerShape: number[]): void;
        feedForward(input: Layer, isInTraining: boolean): void;
        backPropagation(prev_layer: Layer, next_layer: Layer | Matrix): void;
        toSavedModel(): SavedLayer;
        fromSavedModel(data: SavedLayer): void;
    }
}
declare module "losses/gradients" {
    import { IActivation } from "activations/activations";
    import { ILoss } from "losses/losses";
    import Matrix from "matrix";
    export default class Gradients {
        static get_gradient(actvFunc: IActivation, lossFunc: ILoss): IGradient;
    }
    export interface IGradient {
        (input: Matrix, labels: Matrix): Matrix;
    }
}
declare module "layers/output_layer" {
    import Layer from "layers/layer";
    import Matrix from "matrix";
    import DenseLayer from "layers/dense_layer";
    import { IActivation } from "activations/activations";
    import { ILoss } from "losses/losses";
    import { SavedLayer } from "model";
    import { IGradient } from "losses/gradients";
    export default class OutputLayer extends DenseLayer {
        loss: number;
        accuracy: number;
        layerSize: number;
        lossFunction: ILoss;
        gradientFunction: IGradient;
        constructor(layerSize?: number, activation?: IActivation);
        buildLayer(prevLayerShape: number[]): void;
        backPropagationOutputLayer(labels: Matrix, next_layer: Layer): void;
        toSavedModel(): SavedLayer;
        fromSavedModel(data: SavedLayer): void;
    }
}
declare module "layers/pooling_layer" {
    import Layer from "layers/layer";
    import Tensor from "tensor";
    import { SavedLayer } from "model";
    export default class PoolingLayer extends Layer {
        type: string;
        prevShape: number[];
        filterSize: number[];
        padding: number;
        stride: number[];
        channel_first: boolean;
        poolingFunc: string;
        constructor(filterSize?: number[], stride?: number[], ch_first?: boolean);
        buildLayer(prevLayerShape: number[]): void;
        feedForward(input: Layer, isInTraining: boolean): void;
        backPropagation(prev_layer: Layer, next_layer: Layer | Tensor[]): void;
        toSavedModel(): SavedLayer;
        fromSavedModel(data: SavedLayer): void;
    }
}
declare module "layers/layer_helper" {
    import ConvolutionLayer from "layers/conv_layer";
    import DenseLayer from "layers/dense_layer";
    import DropoutLayer from "layers/dropout_layer";
    import FlattenLayer from "layers/flatten_layer";
    import PoolingLayer from "layers/pooling_layer";
    export class LayerHelper {
        static fromType(type: string): ConvolutionLayer | DenseLayer | DropoutLayer | FlattenLayer | PoolingLayer;
    }
}
declare module "helpers/helper" {
    export default class Helper {
        static timeit(func: Function, floorIt?: boolean): Promise<number>;
    }
}
declare module "model" {
    import Dataset, { Example } from "dataset";
    import Layer from "layers/layer";
    import Matrix from "matrix";
    import Vector from "vector";
    import { GPU } from 'gpu.js';
    import { ILoss } from "losses/losses";
    import Tensor from "tensor";
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
}
declare module "helpers/matrix_helper" {
    import Matrix from "matrix";
    import Vector from "vector";
    export default class MatrixHelper {
        static row_reduction(matrix: Matrix): Matrix;
        static linear_least_squares(x: Vector, y: Vector): void;
        static linear_least_squares_pol(x: Vector, y: Vector): void;
    }
}
declare module "layers/batch_norm_layer" {
    import Layer from "layers/layer";
    import Matrix from "matrix";
    export default class BatchNormLayer extends Layer {
        momentum: number;
        running_mean: Matrix;
        running_var: Matrix;
        cache: any;
        constructor(momentum?: number);
        buildLayer(prevLayerShape: number[]): void;
        feedForward(input: Layer | Matrix, isInTraining: boolean): void;
        backPropagation(prev_layer: Layer, next_layer: Layer | Matrix): void;
        updateWeights(l_rate: number): void;
    }
}
declare module "linguistics/csv_parser" {
    export default class CsvParser {
        static parse(data: string, isPath?: boolean): (string | number)[][];
        static filterColumns(data: (number | string)[][], columns: number[]): (string | number)[][];
    }
}
declare module "linguistics/suffixes" {
    export const suffixes: string[];
}
declare module "linguistics/tokenizer" {
    export default class Tokenizer {
        vocab: any;
        constructor();
        createVocabulary(sentences: string[]): void;
        loadVocabulary(path: string): void;
        saveVocabulary(path: string): void;
        tokenize(sentence: string): any[];
    }
}
declare module "visualizer/data_handler" {
    import { PubSub } from "apollo-server";
    import { BacklogData } from "model";
    export default class DataHandler {
        pubSub: PubSub;
        watchPath: string;
        data: BacklogData;
        constructor(pubSub: PubSub, path: string);
        loadData(): void;
        startWatcher(): void;
        getBatches(): any[];
        getBatch(epoch_id: number, batch_id: number): any;
        private parseEpoch;
        getEpochs(): any[];
        getEpoch(epoch_id: number): any;
    }
}
declare module "visualizer/visualizer" {
    import { ApolloServer } from "apollo-server";
    import DataHandler from "visualizer/data_handler";
    export default class Visualizer {
        PORT: number;
        pubsub: any;
        data_handler: DataHandler;
        server: ApolloServer;
        constructor(path: string);
        run(): void;
    }
}
