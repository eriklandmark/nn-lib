"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var vector_1 = __importDefault(require("./vector"));
var Activations = /** @class */ (function () {
    function Activations() {
    }
    Activations.lookUp = function (name) {
        switch (name) {
            case "sigmoid": return { func: Activations.sigmoid, derv: Activations.sigmoid_derivative };
            case "relu": return { func: Activations.ReLu, derv: Activations.ReLu_derivative };
            case "softmax": return { func: Activations.Softmax, derv: null };
            default: return { func: Activations.sigmoid, derv: Activations.sigmoid_derivative };
        }
    };
    Activations.sigmoid_gpu = function () {
        return function actv(a) {
            return (1 / (1 + Math.exp(-a)));
        };
    };
    Activations.sigmoid = function (v) {
        if (v instanceof vector_1.default) {
            v.iterate(function (val, i) {
                v.set(i, (1 / (1 + Math.exp(-val))));
            });
            return v;
        }
        else {
            var m_1 = v.copy();
            m_1.iterate(function (i, j) {
                m_1.set(i, j, 1 / (1 + Math.exp(-v.get(i, j))));
            });
            return m_1;
        }
    };
    Activations.sigmoid_derivative = function (v) {
        if (v instanceof vector_1.default) {
            v.iterate(function (val, i) {
                v.set(i, val * (1 - val));
            });
            return v;
        }
        else {
            var m_2 = v.copy();
            m_2.iterate(function (i, j) {
                m_2.set(i, j, m_2.get(i, j) * (1 - m_2.get(i, j)));
            });
            return m_2;
        }
    };
    Activations.ReLu = function (v) {
        if (v instanceof vector_1.default) {
            v.iterate(function (val, i) {
                v.set(i, Math.max(val, 0));
            });
            return v;
        }
        else {
            var m_3 = v.copy();
            m_3.iterate(function (i, j) {
                m_3.set(i, j, Math.max(m_3.get(i, j), 0));
            });
            return m_3;
        }
    };
    Activations.ReLu_derivative = function (v) {
        if (v instanceof vector_1.default) {
            v.iterate(function (val, i) {
                v.set(i, val > 0 ? 1 : 0);
            });
            return v;
        }
        else {
            var m_4 = v.copy();
            m_4.iterate(function (i, j) {
                m_4.set(i, j, m_4.get(i, j) > 0 ? 1 : 0);
            });
            return m_4;
        }
    };
    Activations.Softmax = function (m) {
        var exp = m.exp();
        return exp.div(exp.sum(1, true));
    };
    return Activations;
}());
exports.default = Activations;
