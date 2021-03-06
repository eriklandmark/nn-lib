import ConvolutionLayer from "./conv_layer";
import DenseLayer from "./dense_layer";
import DropoutLayer from "./dropout_layer";
import FlattenLayer from "./flatten_layer";
import PoolingLayer from "./pooling_layer";
export declare class LayerHelper {
    static fromType(type: string): ConvolutionLayer | DenseLayer | DropoutLayer | FlattenLayer | PoolingLayer;
}
