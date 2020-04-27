"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const apollo_server_1 = require("apollo-server");
const graphql_tools_1 = require("graphql-tools");
const data_handler_1 = __importDefault(require("./data_handler"));
class Visualizer {
    constructor(path) {
        this.PORT = 3000;
        this.pubsub = new apollo_server_1.PubSub();
        this.data_handler = new data_handler_1.default(this.pubsub, path);
        const typeDefs = apollo_server_1.gql(`
              type Epoch {
                id: Float
                accuracy: Float
                total_accuracy: Float
                loss: Float
                total_loss: Float
                actual_duration: Float
                calculated_duration: Float
                batches: [Batch]
              }
              
              type Batch {
                id: Float
                epoch_id: Float
                accuracy: Float
                loss: Float
                time: Float
              }
              
              type Query {
                epochs:[Epoch]
                epoch(id: Float): Epoch
                batches: [Batch]
                batch(id: Float, epoch_id: Float): Batch
              }
              
              type Subscription {
                new_batch: Batch
              }
            `);
        const schema = graphql_tools_1.makeExecutableSchema({
            typeDefs, resolvers: {
                Query: {
                    batches: () => {
                        return this.data_handler.getBatches();
                    },
                    batch: (parent, args, context, info) => {
                        return this.data_handler.getBatch(args.epoch_id, args.id);
                    },
                    epochs: () => {
                        return this.data_handler.getEpochs();
                    },
                    epoch: (parent, args, context, info) => {
                        return this.data_handler.getEpoch(args.id);
                    }
                },
                Subscription: {
                    new_batch: {
                        subscribe: () => this.pubsub.asyncIterator("new_batch"),
                    },
                }
            }
        });
        this.server = new apollo_server_1.ApolloServer({ schema });
    }
    run() {
        console.log("Starting server...");
        this.data_handler.startWatcher();
        this.server.listen({ port: this.PORT }).then(({ url }) => {
            console.log(`Visualizer server ready at ${url}`);
        });
    }
}
exports.default = Visualizer;
