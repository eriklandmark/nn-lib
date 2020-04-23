import {PubSub} from "apollo-server"
import fs from "fs"
import {BacklogData} from "../model";

export default class DataHandler {

    pubSub: PubSub
    watchPath: string
    data: BacklogData

    constructor(pubSub: PubSub, path: string) {
        this.pubSub = pubSub
        this.watchPath = path
        this.loadData()
    }

    loadData() {
        this.data = JSON.parse(fs.readFileSync(this.watchPath, {encoding: "utf-8"}))
    }

    startWatcher() {
        if(!fs.existsSync(this.watchPath)) {
            throw "Backlog file doesn't exists... Aborting!"
        }
        fs.watchFile(this.watchPath, {}, (stats) => {
            console.log("Backlog updated.")
            this.loadData()
        })
        console.log("Started backlog watcher!")
    }

    getBatches() {
        return Object.keys(this.data.epochs).reduce((acc, epoch) => {
            acc.push(...this.data.epochs[epoch].batches.map((batch, index) => {
                batch["id"] = index
                batch["epoch_id"] = parseInt(epoch.substr(epoch.lastIndexOf("_") + 1));
                return batch
            }))
            return acc
        }, [])
    }

    getBatch(epoch_id: number, batch_id: number) {
        return this.getBatches().filter((batch) => batch.id == batch_id && batch.epoch_id == epoch_id)[0]
    }

    private parseEpoch(epoch: string) {
        const epoch_id = parseInt(epoch.substr(epoch.lastIndexOf("_") + 1))
        const data = Object.create(this.data.epochs[epoch])
        data.batches = data.batches.map((batch, index) => {
            batch["id"] = index
            batch["epoch_id"] = epoch_id;
            return batch
        })
        data["accuracy"] = this.data.epochs[epoch].total_accuracy / this.data.epochs[epoch].batches.length
        data["loss"] = this.data.epochs[epoch].total_loss / this.data.epochs[epoch].batches.length
        data["id"] = epoch_id
        return data
    }

    getEpochs() {
        return Object.keys(this.data.epochs).map((epoch) => {
            return this.parseEpoch(epoch)
        })
    }

    getEpoch(epoch_id: number) {
        return this.parseEpoch("epoch_" + epoch_id)
    }
}