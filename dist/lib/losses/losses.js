"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var vector_1 = __importDefault(require("../../vector"));
var cross_entropy_1 = __importDefault(require("./cross_entropy"));
var mean_squared_error_1 = __importDefault(require("./mean_squared_error"));
var Losses = /** @class */ (function () {
    function Losses() {
    }
    Losses.fromName = function (name) {
        switch (name) {
            case "cross_entropy": return new cross_entropy_1.default();
            case "mean_squared_error": return new mean_squared_error_1.default();
        }
    };
    Losses.CrossEntropy = function (v, labels) {
        var out = new vector_1.default(v.size());
        if (v.size() != labels.size())
            throw "Labels and output vector doesn't match size..";
        out.iterate(function (_, i) {
            var a = v.get(i);
            var y = labels.get(i);
            if (a == 1) {
                out.set(i, -Math.log(a));
            }
            else {
                out.set(i, -1 * ((y * Math.log(a)) + (1 - y) * Math.log(1 - a)));
            }
        });
        return out;
    };
    Losses.CrossEntropy_derivative = function (v, labels) {
        var out = new vector_1.default(v.size());
        if (v.size() != labels.size())
            throw "Labels and output vector doesn't match size..";
        out.iterate(function (_, i) {
            var a = v.get(i);
            var y = labels.get(i);
            out.set(i, (-y / a) + ((1 - y) / (1 - a)));
        });
        return out;
    };
    return Losses;
}());
exports.default = Losses;
