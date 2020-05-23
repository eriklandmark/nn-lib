"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const tensor_1 = __importDefault(require("../tensor"));
class Sigmoid {
    constructor() {
        this.name = "sigmoid";
    }
    normal_gpu() {
        return function actv(a) {
            return (1 / (1 + Math.exp(-a[this.thread.y][this.thread.x])));
        };
    }
    derivative_gpu() {
        return function actv_der(a) {
            return a * (1 - a);
        };
    }
    normal(input) {
        if (input instanceof tensor_1.default) {
            const m = input.copy(false);
            m.iterate((pos) => {
                m.set(pos, 1 / (1 + Math.exp(-input.get(pos))));
            }, true);
            return m;
        }
        else {
            return 1 / (1 + Math.exp(-input));
        }
    }
    derivative(input) {
        if (input instanceof tensor_1.default) {
            const m = input.copy(false);
            m.iterate((pos) => {
                m.set(pos, input.get(pos) * (1 - input.get(pos)));
            }, true);
            return m;
        }
        else {
            return input * (1 - input);
        }
    }
}
exports.default = Sigmoid;
