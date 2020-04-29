"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const dense_layer_1 = __importDefault(require("./dense_layer"));
const losses_1 = __importDefault(require("../losses/losses"));
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
        this.loss = labels.mul(-1).mul(this.activation.add(Math.pow(10, -8)).log()).sum();
        const gradient = this.gradientFunction(this.activation, labels);
        let total_acc = 0;
        let total_loss = 0;
        for (let i = 0; i < labels.dim().r; i++) {
            total_acc += this.activation.argmax(i) == labels.argmax(i) ? 1 : 0;
            total_loss += Math.abs(gradient.get(i, 0));
        }
        this.accuracy = total_acc / labels.dim().r;
        //this.loss = total_loss
        this.errorBias = gradient.sum(0, false);
        this.output_error = gradient;
        this.errorWeights = next_layer.activation.transpose().mm(gradient);
    }
    toSavedModel() {
        const data = super.toSavedModel();
        data.layer_specific = {
            loss: this.lossFunction.name
        };
        return data;
    }
    fromSavedModel(data) {
        super.fromSavedModel(data);
        this.lossFunction = losses_1.default.fromName(data.layer_specific.loss);
    }
}
exports.default = OutputLayer;
