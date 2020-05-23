import ModelPool, {ModelConfig} from "../src/model_pool";
import ConvolutionLayer from "../src/layers/conv_layer";
import ReLu from "../src/activations/relu";
import PoolingLayer from "../src/layers/pooling_layer";
import FlattenLayer from "../src/layers/flatten_layer";
import DenseLayer from "../src/layers/dense_layer";
import OutputLayer from "../src/layers/output_layer";
import Softmax from "../src/activations/softmax";
import HyperbolicTangent from "../src/activations/hyperbolic_tangent";

const SAVE_PATH = "./models/model_test3"
const EVAL = true

const modelPool = new ModelPool("./test/train_node.ts", [
    {
        layers: [
            new ConvolutionLayer(8, [5, 5], false, new ReLu()),
            new PoolingLayer([2, 2], [2, 2]),
            new ConvolutionLayer(16, [5, 5], false, new ReLu()),
            new FlattenLayer(),
            new DenseLayer(700, new HyperbolicTangent()),
            new OutputLayer(10, new Softmax())
        ],
        settings: {
            USE_GPU: false,
            MODEL_SAVE_PATH: SAVE_PATH,
            BACKLOG: true,
            EVAL_PER_EPOCH: EVAL,
            SAVE_CHECKPOINTS: true,
            VERBOSE_COMPACT: true
        },
        train_settings: {
            shuffle: true,
            epochs: 80,
            learning_rate: 0.01,
            optimizer: "adam",
            loss: "cross_entropy"
        }
    } as ModelConfig,
    {
        layers: [
            new ConvolutionLayer(8, [5, 5], false, new ReLu()),
            new PoolingLayer([2, 2], [2, 2]),
            new ConvolutionLayer(16, [5, 5], false, new ReLu()),
            new FlattenLayer(),
            new DenseLayer(500, new HyperbolicTangent()),
            new OutputLayer(10, new Softmax())
        ],
        settings: {
            USE_GPU: false,
            MODEL_SAVE_PATH: SAVE_PATH,
            BACKLOG: true,
            EVAL_PER_EPOCH: EVAL,
            SAVE_CHECKPOINTS: true,
            VERBOSE_COMPACT: true
        },
        train_settings: {
            shuffle: true,
            epochs: 80,
            learning_rate: 0.01,
            optimizer: "adam",
            loss: "cross_entropy"
        }
    } as ModelConfig,
    {
        layers: [
            new ConvolutionLayer(8, [5, 5], false, new ReLu()),
            new PoolingLayer([2, 2], [2, 2]),
            new ConvolutionLayer(16, [5, 5], false, new ReLu()),
            new FlattenLayer(),
            new DenseLayer(300, new HyperbolicTangent()),
            new OutputLayer(10, new Softmax())
        ],
        settings: {
            USE_GPU: false,
            MODEL_SAVE_PATH: SAVE_PATH,
            BACKLOG: true,
            EVAL_PER_EPOCH: EVAL,
            SAVE_CHECKPOINTS: true,
            VERBOSE_COMPACT: true
        },
        train_settings: {
            shuffle: true,
            epochs: 80,
            learning_rate: 0.01,
            optimizer: "adam",
            loss: "cross_entropy"
        }
    } as ModelConfig,
    {
        layers: [
            new ConvolutionLayer(8, [5, 5], false, new ReLu()),
            new PoolingLayer([2, 2], [2, 2]),
            new ConvolutionLayer(16, [5, 5], false, new ReLu()),
            new FlattenLayer(),
            new DenseLayer(700, new HyperbolicTangent()),
            new OutputLayer(10, new Softmax())
        ],
        settings: {
            USE_GPU: false,
            MODEL_SAVE_PATH: SAVE_PATH,
            BACKLOG: true,
            EVAL_PER_EPOCH: EVAL,
            SAVE_CHECKPOINTS: true,
            VERBOSE_COMPACT: true
        },
        train_settings: {
            shuffle: true,
            epochs: 80,
            learning_rate: 0.005,
            optimizer: "adam",
            loss: "cross_entropy"
        }
    } as ModelConfig,
    {
        layers: [
            new ConvolutionLayer(8, [5, 5], false, new ReLu()),
            new PoolingLayer([2, 2], [2, 2]),
            new ConvolutionLayer(16, [5, 5], false, new ReLu()),
            new FlattenLayer(),
            new DenseLayer(500, new HyperbolicTangent()),
            new OutputLayer(10, new Softmax())
        ],
        settings: {
            USE_GPU: false,
            MODEL_SAVE_PATH: SAVE_PATH,
            BACKLOG: true,
            EVAL_PER_EPOCH: EVAL,
            SAVE_CHECKPOINTS: true,
            VERBOSE_COMPACT: true
        },
        train_settings: {
            shuffle: true,
            epochs: 80,
            learning_rate: 0.005,
            optimizer: "adam",
            loss: "cross_entropy"
        }
    } as ModelConfig,
    {
        layers: [
            new ConvolutionLayer(8, [5, 5], false, new ReLu()),
            new PoolingLayer([2, 2], [2, 2]),
            new ConvolutionLayer(16, [5, 5], false, new ReLu()),
            new FlattenLayer(),
            new DenseLayer(300, new HyperbolicTangent()),
            new OutputLayer(10, new Softmax())
        ],
        settings: {
            USE_GPU: false,
            MODEL_SAVE_PATH: SAVE_PATH,
            BACKLOG: true,
            EVAL_PER_EPOCH: EVAL,
            SAVE_CHECKPOINTS: true,
            VERBOSE_COMPACT: true
        },
        train_settings: {
            shuffle: true,
            epochs: 80,
            learning_rate: 0.005,
            optimizer: "adam",
            loss: "cross_entropy"
        }
    } as ModelConfig
])

modelPool.run()