"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var MeanSquaredError = /** @class */ (function () {
    function MeanSquaredError() {
        this.name = "mean_squared_error";
    }
    MeanSquaredError.prototype.normal = function (input, labels) {
        if (input.dim().r != labels.dim().r || input.dim().c != labels.dim().c)
            throw "Labels and output vector doesn't match size..";
        return input.sub(labels).pow(2);
    };
    MeanSquaredError.prototype.derivative = function (input, labels) {
        if (input.dim().r != labels.dim().r || input.dim().c != labels.dim().c)
            throw "Labels and output vector doesn't match size..";
        return input.sub(labels);
    };
    MeanSquaredError.prototype.normal_gpu = function () {
        return function actv() { };
    };
    MeanSquaredError.prototype.derivative_gpu = function () {
        return function loss(m, label) {
            //@ts-ignore
            return m - label;
        };
    };
    return MeanSquaredError;
}());
exports.default = MeanSquaredError;
