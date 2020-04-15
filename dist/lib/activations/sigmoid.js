"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var matrix_1 = __importDefault(require("../../matrix"));
var Sigmoid = /** @class */ (function () {
    function Sigmoid() {
        this.name = "sigmoid";
    }
    Sigmoid.prototype.normal_gpu = function () {
        return function actv(a) {
            return (1 / (1 + Math.exp(-a[this.thread.y][this.thread.x])));
        };
    };
    Sigmoid.prototype.derivative_gpu = function () {
        return function actv_der(a) {
            return a * (1 - a);
        };
    };
    Sigmoid.prototype.normal = function (input) {
        if (input instanceof matrix_1.default) {
            var m_1 = input.copy();
            m_1.iterate(function (i, j) {
                m_1.set(i, j, 1 / (1 + Math.exp(-input.get(i, j))));
            });
            return m_1;
        }
        else {
            return 1 / (1 + Math.exp(-input));
        }
    };
    Sigmoid.prototype.derivative = function (input) {
        if (input instanceof matrix_1.default) {
            var m_2 = input.copy();
            m_2.iterate(function (i, j) {
                m_2.set(i, j, input.get(i, j) * (1 - input.get(i, j)));
            });
            return m_2;
        }
        else {
            return input * (1 - input);
        }
    };
    return Sigmoid;
}());
exports.default = Sigmoid;
