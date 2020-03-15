"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ReLu = /** @class */ (function () {
    function ReLu() {
        this.name = "relu";
    }
    ReLu.prototype.normal = function (input) {
        var m = input.copy();
        m.iterate(function (i, j) {
            m.set(i, j, Math.max(m.get(i, j), 0));
        });
        return m;
    };
    ReLu.prototype.derivative = function (input) {
        var m = input.copy();
        m.iterate(function (i, j) {
            m.set(i, j, m.get(i, j) > 0 ? 1 : 0);
        });
        return m;
    };
    ReLu.prototype.normal_gpu = function () {
        return function actv() { };
    };
    ReLu.prototype.derivative_gpu = function () {
        return function actv() { };
    };
    return ReLu;
}());
exports.default = ReLu;
