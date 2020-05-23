"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const tensor_1 = __importDefault(require("../tensor"));
class ReLu {
    constructor() {
        this.name = "relu";
    }
    normal(input) {
        if (input instanceof tensor_1.default) {
            const m = input.copy(false);
            m.iterate((pos) => {
                m.set(pos, Math.max(input.get(pos), 0));
            }, true);
            return m;
        }
        else {
            return Math.max(input, 0);
        }
    }
    derivative(input) {
        if (input instanceof tensor_1.default) {
            const m = input.copy(false);
            m.iterate((pos) => {
                m.set(pos, input.get(pos) > 0 ? 1 : 0);
            }, true);
            return m;
        }
        else {
            return input > 0 ? 1 : 0;
        }
    }
    normal_gpu() {
        return function actv(a) {
            return Math.max(a[this.thread.y][this.thread.x], 0);
        };
    }
    derivative_gpu() {
        return function actv(a) {
            return a > 0 ? 1 : 0;
        };
    }
}
exports.default = ReLu;
