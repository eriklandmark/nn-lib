"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const model_1 = __importDefault(require("./model"));
const layer_helper_1 = require("./layers/layer_helper");
const losses_1 = __importDefault(require("./losses/losses"));
const Optimizers_1 = __importDefault(require("./optimizers/Optimizers"));
class ModelNode {
    constructor(dataset, eval_dataset = null) {
        process.on("message", (data) => {
            if (data["action"] == "run") {
                const config = data["model"];
                this.train_settings = config.train_settings;
                const layers = config.layers.map((layer_conf) => {
                    let layer = layer_helper_1.LayerHelper.fromType(layer_conf.type);
                    layer.fromSavedModel(layer_conf);
                    return layer;
                });
                this.model = new model_1.default(layers);
                this.model.settings = config.settings;
                this.model.settings.MODEL_SAVE_PATH = this.model.settings.MODEL_SAVE_PATH + "_" + data["id"];
                this.model.settings.WORKER_MODE = true;
                this.model.settings.WORKER_CALLBACK = (cData) => {
                    cData["id"] = data["id"];
                    process.send({ action: "update", data: cData });
                };
                this.model.build(dataset.DATA_SHAPE, this.train_settings.learning_rate, losses_1.default.fromName(this.train_settings.loss), Optimizers_1.default.fromName(this.train_settings.optimizer), false);
                process.send({
                    action: "built",
                    total_steps: ((dataset.TOTAL_EXAMPLES / dataset.BATCH_SIZE) * this.train_settings.epochs)
                });
                this.model.train(dataset, this.train_settings.epochs, eval_dataset, this.train_settings.shuffle).then(() => {
                    this.model.save();
                });
            }
        });
    }
}
exports.default = ModelNode;
