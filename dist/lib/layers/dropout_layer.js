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
var DropoutLayer = /** @class */ (function (_super) {
    __extends(DropoutLayer, _super);
    function DropoutLayer(rate) {
        if (rate === void 0) { rate = 0.2; }
        var _this = _super.call(this) || this;
        _this.rate = 0;
        _this.type = "dropout";
        _this.rate = rate;
        return _this;
    }
    DropoutLayer.prototype.buildLayer = function (prevLayerShape) {
        this.shape = prevLayerShape;
    };
    DropoutLayer.prototype.feedForward = function (input, isInTraining) {
        var _this = this;
        this.activation = input.activation;
        if (isInTraining) {
            this.activation.iterate(function (i, j) {
                if (Math.random() < _this.rate) {
                    _this.activation.set(i, j, 0);
                }
            });
        }
    };
    DropoutLayer.prototype.backPropagation = function (prev_layer, next_layer) {
        this.weights = prev_layer.weights;
        this.output_error = prev_layer.output_error;
    };
    DropoutLayer.prototype.updateWeights = function (l_rate) { };
    DropoutLayer.prototype.toSavedModel = function () {
        return {
            rate: this.rate,
            shape: this.shape
        };
    };
    DropoutLayer.prototype.fromSavedModel = function (data) {
        this.shape = data.shape;
        this.rate = data.rate;
    };
    return DropoutLayer;
}(layer_1.default));
exports.default = DropoutLayer;
