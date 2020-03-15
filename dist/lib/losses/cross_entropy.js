"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var CrossEntropy = /** @class */ (function () {
    function CrossEntropy() {
        this.name = "cross_entropy";
    }
    CrossEntropy.prototype.normal = function (input) {
        var exp = input.exp();
        return exp.div(exp.sum(1, true));
    };
    CrossEntropy.prototype.derivative = function (input) {
        return input;
    };
    CrossEntropy.prototype.normal_gpu = function () {
        return function actv() { };
    };
    CrossEntropy.prototype.derivative_gpu = function () {
        return function actv() { };
    };
    return CrossEntropy;
}());
exports.default = CrossEntropy;
