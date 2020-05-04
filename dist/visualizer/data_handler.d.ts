import { PubSub } from "apollo-server";
import { BacklogData } from "../model";
export default class DataHandler {
    pubSub: PubSub;
    watchPath: string;
    data: BacklogData;
    constructor(pubSub: PubSub, path: string);
    loadData(): void;
    startWatcher(): void;
    getModelInfo(): {
        model_structure: any;
        total_neurons: number;
        duration: number;
        start_time: number;
        total_epochs: number;
        batches_per_epoch: number;
        eval_model: boolean;
    };
    getBatches(): any[];
    getBatch(epoch_id: number, batch_id: number): any;
    private parseEpoch;
    getEpochs(): any[];
    getEpoch(epoch_id: number): any;
}
