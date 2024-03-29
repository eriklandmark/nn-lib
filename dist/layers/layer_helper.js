import ConvolutionLayer from "./conv_layer";
import DenseLayer from "./dense_layer";
import DropoutLayer from "./dropout_layer";
import FlattenLayer from "./flatten_layer";
import OutputLayer from "./output_layer";
import Sigmoid from "../activations/sigmoid";
import PoolingLayer from "./pooling_layer";
export class LayerHelper {
    static fromType(type) {
        switch (type) {
            case "conv": return new ConvolutionLayer(0, [], false, new Sigmoid());
            case "dense": return new DenseLayer();
            case "dropout": return new DropoutLayer();
            case "flatten": return new FlattenLayer();
            case "output": return new OutputLayer();
            case "pooling": return new PoolingLayer();
        }
    }
}
