"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var matrix_1 = __importDefault(require("../../matrix"));
var Softmax = /** @class */ (function () {
    function Softmax() {
        this.name = "softmax";
    }
    Softmax.prototype.normal = function (input) {
        var exp = input.exp();
        return exp.div(exp.sum(1, true));
    };
    Softmax.prototype.derivative = function (input) {
        if (input instanceof matrix_1.default) {
            var m_1 = input.copy();
            m_1.iterate(function (i, j) {
                if (i == j) {
                    m_1.set(i, j, input.get(i, j) * (1 - input.get(i, j)));
                }
                else {
                    m_1.set(i, j, -(input.get(i, j) * input.get(i, j)));
                }
            });
            return m_1;
        }
        else {
            return input * (1 - input);
        }
    };
    Softmax.prototype.normal_gpu = function () {
        return function actv(a) {
            var sum = 0;
            for (var i = 0; i < this.constants.softmax; i++) {
                sum += Math.exp(a[this.thread.y][i]);
            }
            return Math.exp(a[this.thread.y][this.thread.x]) / sum;
        };
    };
    Softmax.prototype.derivative_gpu = function () {
        return function actv_der(a) {
            return a * (1 - a);
        };
    };
    return Softmax;
}());
exports.default = Softmax;
