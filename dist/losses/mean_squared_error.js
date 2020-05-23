"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
class MeanSquaredError {
    constructor() {
        this.name = "mean_squared_error";
    }
    normal(input, labels) {
        if (!input.equalShape(labels))
            throw "Labels and output vector doesn't match size..";
        return input.sub(labels).pow(2);
    }
    derivative(input, labels) {
        if (!input.equalShape(labels))
            throw "Labels and output vector doesn't match size..";
        return input.sub(labels);
    }
    normal_gpu() {
        return function actv() {
            return;
        };
    }
    derivative_gpu() {
        return function loss(m, label) {
            //@ts-ignore
            return m - label;
        };
    }
}
exports.default = MeanSquaredError;
