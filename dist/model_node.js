import Model from "./model";
import { LayerHelper } from "./layers/layer_helper";
import Losses from "./losses/losses";
import Optimizers from "./optimizers/Optimizers";
export default class ModelNode {
    constructor(dataset, eval_dataset = null) {
        process.on("message", (data) => {
            if (data["action"] == "run") {
                const config = data["model"];
                this.train_settings = config.train_settings;
                const layers = config.layers.map((layer_conf) => {
                    let layer = LayerHelper.fromType(layer_conf.type);
                    layer.fromSavedModel(layer_conf);
                    return layer;
                });
                this.model = new Model(layers);
                this.model.settings = config.settings;
                this.model.settings.MODEL_SAVE_PATH = this.model.settings.MODEL_SAVE_PATH + "_" + data["id"];
                this.model.settings.WORKER_MODE = true;
                this.model.settings.WORKER_CALLBACK = (cData) => {
                    cData["id"] = data["id"];
                    process.send({ action: "update", data: cData });
                };
                this.model.build(dataset.DATA_SHAPE, this.train_settings.learning_rate, Losses.fromName(this.train_settings.loss), Optimizers.fromName(this.train_settings.optimizer), false);
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
