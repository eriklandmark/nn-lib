"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var matrix_1 = __importDefault(require("../../matrix"));
var ReLu = /** @class */ (function () {
    function ReLu() {
        this.name = "relu";
    }
    ReLu.prototype.normal = function (input) {
        if (input instanceof matrix_1.default) {
            var m_1 = input.copy();
            m_1.iterate(function (i, j) {
                m_1.set(i, j, Math.max(m_1.get(i, j), 0));
            });
            return m_1;
        }
        else {
            return Math.max(input, 0);
        }
    };
    ReLu.prototype.derivative = function (input) {
        if (input instanceof matrix_1.default) {
            var m_2 = input.copy();
            m_2.iterate(function (i, j) {
                m_2.set(i, j, m_2.get(i, j) > 0 ? 1 : 0);
            });
            return m_2;
        }
        else {
            return input > 0 ? 1 : 0;
        }
    };
    ReLu.prototype.normal_gpu = function () {
        return function actv(a) { };
    };
    ReLu.prototype.derivative_gpu = function () {
        return function actv(a) {
            return a;
        };
    };
    return ReLu;
}());
exports.default = ReLu;
