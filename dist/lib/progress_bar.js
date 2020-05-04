"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const cli_progress_1 = __importDefault(require("cli-progress"));
class ProgressBar {
    constructor(format, total, data, fps = 10) {
        this.bar = new cli_progress_1.default.Bar({
            barCompleteChar: '#',
            barIncompleteChar: '-',
            format: format,
            fps: fps,
            stream: process.stdout,
            barsize: 15
        });
        this.total = total;
        this.start_data = data;
    }
    start() {
        this.bar.start(this.total, 0, this.start_data);
    }
    increment(data = {}) {
        this.bar.increment(1, data);
    }
    stop() {
        this.bar.stop();
    }
}
exports.default = ProgressBar;
