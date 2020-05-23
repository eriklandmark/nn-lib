import Layer from "./layers/layer";
import { ModelSettings, SavedLayer } from "./model";
export interface ModelConfig {
    layers: Layer[] | SavedLayer[];
    settings: ModelSettings;
    train_settings: {
        epochs: number;
        shuffle: boolean;
        learning_rate: number;
        optimizer: any;
        loss: any;
    };
}
export default class ModelPool {
    models: {
        proc: any;
        config: ModelConfig;
    }[];
    progressBars: any;
    multiBar: any;
    constructor(model_node_path: string, model_configs: ModelConfig[]);
    run(): void;
}
