"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const child_process_1 = __importDefault(require("child_process"));
const cli_progress_1 = __importDefault(require("cli-progress"));
class ModelPool {
    constructor(model_node_path, model_configs) {
        this.progressBars = {};
        this.multiBar = new cli_progress_1.default.MultiBar({
            hideCursor: true,
            barCompleteChar: '#',
            barIncompleteChar: '-',
            format: 'M{model_id} - [{bar}] {percentage}% | E {epoch}/{epoch_tot} | B {batch}/{batch_tot} | {value}' +
                ' | loss: {loss} | acc: {acc} | Time (TOT/AVG): {time_tot} / {time_avg}',
            fps: 15,
            stream: process.stdout,
            barsize: 10
        });
        this.models = model_configs.map((config, index) => {
            const proc = child_process_1.default.fork(model_node_path);
            proc.on('message', (data) => {
                if (data["action"] == "done") {
                    console.log("Done");
                }
                else if (data["action"] == "update") {
                    const id = data["data"]["epoch"] == 1 ? data["data"]["batch"] :
                        (data["data"]["epoch"] - 1) * data["data"]["batch_tot"] + data["data"]["batch"];
                    this.progressBars[data["data"]["id"]].update(id, data["data"]);
                }
                else if (data["action"] == "built") {
                    this.progressBars[index.toString()] = this.multiBar.create(data["total_steps"], 0, {
                        acc: (0).toPrecision(3),
                        time_tot: (0).toPrecision(5),
                        time_avg: (0).toPrecision(5),
                        loss: (0).toPrecision(5),
                        batch: "0",
                        batch_tot: "0",
                        epoch: "0",
                        epoch_tot: "0",
                        model_id: index.toString()
                    });
                }
            });
            return { proc: proc, config: config };
        });
    }
    run() {
        console.log("Building pool..");
        this.models.forEach((model, index) => {
            let config = model.config;
            config.layers = model.config.layers.map((layer) => {
                const data = layer.toSavedModel();
                data.type = layer.type;
                return data;
            });
            model.proc.send({ action: "run", id: index.toString(),
                model: config
            });
        });
    }
}
exports.default = ModelPool;
