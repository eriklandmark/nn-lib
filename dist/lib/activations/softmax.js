"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Softmax = /** @class */ (function () {
    function Softmax() {
        this.name = "softmax";
    }
    Softmax.prototype.normal = function (input) {
        var exp = input.exp();
        return exp.div(exp.sum(1, true));
    };
    Softmax.prototype.derivative = function (input) {
        return input;
    };
    Softmax.prototype.normal_gpu = function () {
        return function actv() { };
    };
    Softmax.prototype.derivative_gpu = function () {
        return function actv() { };
    };
    return Softmax;
}());
exports.default = Softmax;
