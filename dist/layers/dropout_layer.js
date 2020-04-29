"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const layer_1 = __importDefault(require("./layer"));
class DropoutLayer extends layer_1.default {
    constructor(rate = 0.2) {
        super();
        this.rate = 0;
        this.type = "dropout";
        this.rate = rate;
    }
    buildLayer(prevLayerShape) {
        this.shape = prevLayerShape;
    }
    feedForward(input, isInTraining) {
        this.activation = input.activation;
        if (isInTraining) {
            this.activation.iterate((i, j) => {
                if (Math.random() < this.rate) {
                    this.activation.set(i, j, 0);
                }
            });
        }
    }
    backPropagation(prev_layer, next_layer) {
        this.weights = prev_layer.weights;
        this.output_error = prev_layer.output_error;
    }
    toSavedModel() {
        const data = super.toSavedModel();
        data.layer_specific = {
            rate: this.rate
        };
        return data;
    }
    fromSavedModel(data) {
        super.fromSavedModel(data);
        this.rate = data.layer_specific.rate;
    }
}
exports.default = DropoutLayer;
