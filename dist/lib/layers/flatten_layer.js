"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
var layer_1 = __importDefault(require("./layer"));
var tensor_1 = __importDefault(require("../../tensor"));
var matrix_1 = __importDefault(require("../../matrix"));
var FlattenLayer = /** @class */ (function (_super) {
    __extends(FlattenLayer, _super);
    function FlattenLayer() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.type = "flatten";
        _this.prevShape = [];
        return _this;
    }
    FlattenLayer.prototype.buildLayer = function (prevLayerShape) {
        this.prevShape = prevLayerShape;
        this.shape = [prevLayerShape.reduce(function (acc, val) { return acc * val; })];
    };
    FlattenLayer.prototype.feedForward = function (input, isInTraining) {
        var matrix = new matrix_1.default(input.activation.map(function (tensor) { return tensor.vectorize(true); }));
        this.activation = matrix.transpose();
    };
    FlattenLayer.prototype.backPropagation = function (prev_layer, next_layer) {
        var dout = prev_layer.output_error.mm(prev_layer.weights.transpose());
        var t = new Array(prev_layer.output_error.dim().r);
        for (var i = 0; i < t.length; i++) {
            t[i] = new tensor_1.default();
            t[i].createEmptyArray(this.prevShape[0], this.prevShape[1], this.prevShape[2]);
        }
        var _a = this.prevShape, h = _a[0], w = _a[1], d = _a[2];
        dout.iterate(function (n, i) {
            var r = Math.floor(i / (w * d));
            var c = Math.floor(i / (d) - (r * w));
            var g = Math.floor(i - (c * d) - (r * w * d));
            t[n].set(r, c, g, dout.get(n, i));
        });
        this.output_error = t;
    };
    FlattenLayer.prototype.toSavedModel = function () {
        return {
            shape: this.prevShape
        };
    };
    FlattenLayer.prototype.fromSavedModel = function (data) {
        this.buildLayer(data.shape);
    };
    return FlattenLayer;
}(layer_1.default));
exports.default = FlattenLayer;
