"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const matrix_1 = __importDefault(require("../matrix"));
const dense_layer_1 = __importDefault(require("./dense_layer"));
const activations_1 = __importDefault(require("../activations/activations"));
const losses_1 = __importDefault(require("../losses/losses"));
const vector_1 = __importDefault(require("../vector"));
const sigmoid_1 = __importDefault(require("../activations/sigmoid"));
const gradients_1 = __importDefault(require("../losses/gradients"));
class OutputLayer extends dense_layer_1.default {
    constructor(layerSize = 1, activation = new sigmoid_1.default()) {
        super(layerSize, activation);
        this.loss = 0;
        this.accuracy = 0;
        this.layerSize = 0;
        this.layerSize = layerSize;
        this.type = "output";
    }
    buildLayer(prevLayerShape) {
        super.buildLayer(prevLayerShape);
        this.gradientFunction = gradients_1.default.get_gradient(this.activationFunction, this.lossFunction);
    }
    backPropagationOutputLayer(labels, next_layer) {
        this.loss = labels.mul(-1).mul(this.activation.log()).sum();
        const gradient = this.gradientFunction(this.activation, labels);
        let total_acc = 0;
        let total_loss = 0;
        for (let i = 0; i < labels.dim().r; i++) {
            total_acc += this.activation.argmax(i) == labels.argmax(i) ? 1 : 0;
            total_loss += Math.abs(gradient.get(i, 0));
        }
        this.accuracy = total_acc / labels.dim().r;
        //this.loss = total_loss
        this.errorBias = gradient;
        this.output_error = gradient;
        this.errorWeights = next_layer.activation.transpose().mm(gradient);
    }
    toSavedModel() {
        return {
            weights: this.weights.matrix,
            bias: this.bias.vector,
            loss: this.lossFunction.name,
            shape: this.shape,
            activation: this.activationFunction.name
        };
    }
    fromSavedModel(data) {
        this.weights = matrix_1.default.fromJsonObject(data.weights);
        this.bias = vector_1.default.fromJsonObj(data.bias);
        this.shape = data.shape;
        this.activationFunction = activations_1.default.fromName(data.activation);
        this.lossFunction = losses_1.default.fromName(data.loss);
    }
}
exports.default = OutputLayer;
