import { PubSub } from "apollo-server-express";
import DataHandler from "./data_handler";
export default class Visualizer {
    PORT: number;
    pubsub: PubSub;
    data_handler: DataHandler;
    server: any;
    wsServer: any;
    constructor(path: string);
    run(): void;
}
