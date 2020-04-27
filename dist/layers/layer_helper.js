"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const conv_layer_1 = __importDefault(require("./conv_layer"));
const dense_layer_1 = __importDefault(require("./dense_layer"));
const dropout_layer_1 = __importDefault(require("./dropout_layer"));
const flatten_layer_1 = __importDefault(require("./flatten_layer"));
const output_layer_1 = __importDefault(require("./output_layer"));
const sigmoid_1 = __importDefault(require("../activations/sigmoid"));
const pooling_layer_1 = __importDefault(require("./pooling_layer"));
class LayerHelper {
    static fromType(type) {
        switch (type) {
            case "conv": return new conv_layer_1.default(0, [], false, new sigmoid_1.default());
            case "dense": return new dense_layer_1.default();
            case "dropout": return new dropout_layer_1.default();
            case "flatten": return new flatten_layer_1.default();
            case "output": return new output_layer_1.default();
            case "pooling": return new pooling_layer_1.default();
        }
    }
}
exports.LayerHelper = LayerHelper;
