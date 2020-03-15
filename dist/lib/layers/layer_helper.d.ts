import ConvolutionLayer from "./conv_layer";
import DenseLayer from "./dense_layer";
import DropoutLayer from "./dropout_layer";
import FlattenLayer from "./flatten_layer";

export declare class LayerHelper {
    static fromType(type: string): DenseLayer | ConvolutionLayer | DropoutLayer | FlattenLayer;
}
