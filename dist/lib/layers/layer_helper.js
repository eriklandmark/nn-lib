"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var conv_layer_1 = __importDefault(require("./conv_layer"));
var dense_layer_1 = __importDefault(require("./dense_layer"));
var dropout_layer_1 = __importDefault(require("./dropout_layer"));
var flatten_layer_1 = __importDefault(require("./flatten_layer"));
var output_layer_1 = __importDefault(require("./output_layer"));
var sigmoid_1 = __importDefault(require("../activations/sigmoid"));
var LayerHelper = /** @class */ (function () {
    function LayerHelper() {
    }
    LayerHelper.fromType = function (type) {
        switch (type) {
            case "conv": return new conv_layer_1.default(0, [], false, new sigmoid_1.default());
            case "dense": return new dense_layer_1.default();
            case "dropout": return new dropout_layer_1.default();
            case "flatten": return new flatten_layer_1.default();
            case "output": return new output_layer_1.default();
        }
    };
    return LayerHelper;
}());
exports.LayerHelper = LayerHelper;
