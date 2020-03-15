"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var sigmoid_1 = __importDefault(require("./sigmoid"));
var relu_1 = __importDefault(require("./relu"));
var softmax_1 = __importDefault(require("./softmax"));
var Activation = /** @class */ (function () {
    function Activation() {
    }
    Activation.fromName = function (name) {
        switch (name) {
            case "sigmoid": return new sigmoid_1.default();
            case "relu": return new relu_1.default();
            case "softmax": return new softmax_1.default();
            default: return new sigmoid_1.default();
        }
    };
    return Activation;
}());
exports.default = Activation;
