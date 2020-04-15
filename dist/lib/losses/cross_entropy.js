"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var CrossEntropy = /** @class */ (function () {
    function CrossEntropy() {
        this.name = "cross_entropy";
        this.epsilon = Math.pow(10, -14);
    }
    CrossEntropy.prototype.normal = function (input, labels) {
        var _this = this;
        var out = input.copy(false);
        out.iterate(function (i, j) {
            if (labels.get(i, j) != 0) {
                out.set(i, j, (labels.get(i, j) * Math.log(input.get(i, j) + _this.epsilon)));
                //+((1 - labels.get(i,j))*Math.log10(1 - input.get(i,j) + this.epsilon)))
            }
        });
        return out.sum(1, true).mul(-1);
    };
    CrossEntropy.prototype.derivative = function (input, labels) {
        return labels.mul(-1).div(input);
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
