import { ApolloServer, PubSub } from "apollo-server";
import DataHandler from "./data_handler";
export default class Visualizer {
    PORT: number;
    pubsub: PubSub;
    data_handler: DataHandler;
    server: ApolloServer;
    constructor(path: string);
    run(): void;
}
