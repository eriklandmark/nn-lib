"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const apollo_server_express_1 = require("apollo-server-express");
const express_1 = __importDefault(require("express"));
const http_1 = require("http");
const subscriptions_transport_ws_1 = require("subscriptions-transport-ws");
const graphql_1 = require("graphql");
const graphql_tools_1 = require("graphql-tools");
const data_handler_1 = __importDefault(require("./data_handler"));
const path_1 = __importDefault(require("path"));
class Visualizer {
    constructor(path) {
        this.PORT = 3000;
        this.pubsub = new apollo_server_express_1.PubSub();
        this.data_handler = new data_handler_1.default(this.pubsub, path);
        const typeDefs = apollo_server_express_1.gql(`
              type LayerInfo {
                type: String
                activation: String
                shape: [Int]
              }
        
              type Epoch {
                id: Float
                accuracy: Float
                total_accuracy: Float
                loss: Float
                total_loss: Float
                actual_duration: Float
                calculated_duration: Float
                batches: [Batch]
                eval_loss: Float
                eval_accuracy: Float
              }
              
              type Batch {
                id: Float
                epoch_id: Float
                accuracy: Float
                loss: Float
                time: Float
                global_id: Float
              }
              
              type Info {
                model_structure: [LayerInfo]
                total_neurons: Int
                duration: Float
                start_time: Float
                total_epochs: Int
                batches_per_epoch: Int
                eval_model: Boolean
              }
              
              type Query {
                epochs:[Epoch]
                epoch(id: Float): Epoch
                batches: [Batch]
                batch(id: Float, epoch_id: Float): Batch
                info: Info
              }
              
              type UpdateData {
                epoch: Epoch
                info: Info
              }
              
              type Subscription {
                update: UpdateData
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
                    },
                    info: () => {
                        return this.data_handler.getModelInfo();
                    }
                },
                Subscription: {
                    update: {
                        subscribe: () => this.pubsub.asyncIterator("update"),
                    },
                }
            }
        });
        this.wsServer = http_1.createServer((request, response) => {
            response.writeHead(404);
            response.end();
        });
        const apolloServer = new apollo_server_express_1.ApolloServer({ schema });
        const subscriptionServer = subscriptions_transport_ws_1.SubscriptionServer.create({
            schema,
            execute: graphql_1.execute,
            subscribe: graphql_1.subscribe,
        }, {
            server: this.wsServer,
            path: '/graphql',
        });
        const app = express_1.default();
        // @ts-ignore
        apolloServer.applyMiddleware({ app });
        app.use(express_1.default.static(path_1.default.join(__dirname, 'interface')));
        app.get("*", (req, res) => {
            res.sendFile(path_1.default.join(__dirname, 'interface/index.html'));
        });
        this.server = app;
    }
    run() {
        console.log("Starting server...");
        this.data_handler.startWatcher();
        this.wsServer.listen(3001, () => console.log(`Websocket Server is now running on http://localhost:${this.PORT + 1}`));
        this.server.listen({ port: this.PORT }, () => {
            console.log(`Visualizer server ready at http://localhost:${this.PORT}`);
        });
    }
}
exports.default = Visualizer;
