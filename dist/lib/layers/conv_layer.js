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
var ConvolutionLayer = /** @class */ (function (_super) {
    __extends(ConvolutionLayer, _super);
    function ConvolutionLayer(filterSize) {
        var _this = _super.call(this) || this;
        _this.filterSize = [];
        _this.filterSize = filterSize;
        return _this;
    }
    ConvolutionLayer.prototype.buildLayer = function (prevLayerShape) {
        _super.prototype.buildLayer.call(this, prevLayerShape);
    };
    ConvolutionLayer.prototype.feedForward = function (input, isInTraining) {
        this.activation = input.activation;
    };
    ConvolutionLayer.prototype.backPropagation = function (prev_layer, next_layer) {
        this.weights = prev_layer.weights;
        this.output_error = prev_layer.output_error;
    };
    ConvolutionLayer.prototype.updateWeights = function (l_rate) { };
    return ConvolutionLayer;
}(layer_1.default));
exports.default = ConvolutionLayer;
