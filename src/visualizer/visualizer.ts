import {createServer, Server} from 'http';
import {ApolloServer, gql, PubSub} from "apollo-server";
import {SubscriptionServer} from 'subscriptions-transport-ws';
import {DocumentNode, execute, GraphQLSchema, subscribe} from 'graphql';
import {makeExecutableSchema} from "graphql-tools";
import DataHandler from "./data_handler";

export default class Visualizer {
    PORT = 3000

    pubsub = new PubSub();
    data_handler: DataHandler
    server: ApolloServer

    constructor(path: string) {
        this.data_handler = new DataHandler(this.pubsub, path)
        const typeDefs = gql(`
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
                update: Epoch
              }
            `)

        const schema = makeExecutableSchema({
            typeDefs, resolvers: {
                Query: {
                    batches: () => {
                        return this.data_handler.getBatches()
                    },
                    batch: (parent: any, args: any, context: any, info: any) => {
                        return this.data_handler.getBatch(args.epoch_id, args.id);
                    },
                    epochs: () => {
                        return this.data_handler.getEpochs()
                    },
                    epoch: (parent: any, args: any, context: any, info: any) => {
                        return this.data_handler.getEpoch(args.id);
                    }
                },
                Subscription: {
                    update: {
                        subscribe: () => this.pubsub.asyncIterator("update"),
                    },
                }
            }
        })
        this.server = new ApolloServer({schema})
    }

    run() {
        console.log("Starting server...")

        this.data_handler.startWatcher()

        this.server.listen({port: this.PORT}).then(({url}) => {
            console.log(`Visualizer server ready at ${url}`);
        });
    }
}
