export default class ProgressBar {
    bar: any;
    total: number;
    start_data: any;
    constructor(format: string, total: number, data: any, fps?: number);
    start(): void;
    increment(data?: any): void;
    stop(): void;
}
