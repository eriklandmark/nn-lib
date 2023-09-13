import {ModelConfig} from "./model_pool.ts";
import Model, {SavedLayer} from "./model.ts";
import {LayerHelper} from "./layers/layer_helper.ts";
import Losses from "./losses/losses.ts";
import Optimizers from "./optimizers/Optimizers.ts";
import Dataset from "./dataset.ts";

export default class ModelNode {
    model: Model
    train_settings: any

    constructor(dataset: Dataset, eval_dataset: Dataset = null) {
        process.on("message", (data) => {
            if (data["action"] == "run") {
                const config = <ModelConfig> data["model"]
                this.train_settings = config.train_settings

                const layers = (<SavedLayer[]>config.layers).map((layer_conf: SavedLayer) => {
                    let layer = LayerHelper.fromType(layer_conf.type)
                    layer.fromSavedModel(layer_conf)
                    return layer
                })

                this.model = new Model(layers)
                this.model.settings = config.settings
                this.model.settings.MODEL_SAVE_PATH = this.model.settings.MODEL_SAVE_PATH + "_" + data["id"]
                this.model.settings.WORKER_MODE = true
                this.model.settings.WORKER_CALLBACK = (cData) => {
                    cData["id"] = data["id"];
                    process.send({action: "update", data: cData})
                }

                this.model.build(dataset.DATA_SHAPE, this.train_settings.learning_rate,
                    Losses.fromName(this.train_settings.loss), Optimizers.fromName(this.train_settings.optimizer), false)

                process.send({
                    action: "built",
                    total_steps: ((dataset.TOTAL_EXAMPLES / dataset.BATCH_SIZE)* this.train_settings.epochs)
                })

                this.model.train(dataset, this.train_settings.epochs, eval_dataset, this.train_settings.shuffle).then(() => {
                    this.model.save()
                })
            }
        })
    }
}