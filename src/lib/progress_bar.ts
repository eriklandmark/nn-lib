import cliProgress from "cli-progress";

export default class ProgressBar {

    bar: any
    total: number
    start_data: any

    constructor(format: string, total: number, data: any, fps: number = 10) {
        this.bar = new cliProgress.Bar({
            barCompleteChar: '#',
            barIncompleteChar: '-',
            format: format,
            fps: fps,
            stream: process.stdout,
            barsize: 15
        });
        this.total = total
        this.start_data = data
    }

    start() {
        this.bar.start(this.total, 0, this.start_data)
    }

    increment(data: any = {}) {
        this.bar.increment(1, data)
    }

    stop() {
        this.bar.stop()
    }
}