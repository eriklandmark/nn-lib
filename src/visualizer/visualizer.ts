import {ApolloServer, gql, PubSub} from "apollo-server-express";
import express from "express"
import {createServer} from 'http';
import {SubscriptionServer} from 'subscriptions-transport-ws';
import {execute, subscribe} from 'graphql';
import {makeExecutableSchema} from "graphql-tools";
import DataHandler from "./data_handler";
import Path from "path"

export default class Visualizer {
    PORT = 3000

    pubsub = new PubSub();
    data_handler: DataHandler
    server: any
    wsServer: any

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
        this.wsServer = createServer((request, response) => {
            response.writeHead(404)
            response.end()
        });

        const apolloServer = new ApolloServer({schema})

        const subscriptionServer = SubscriptionServer.create(
            {
                schema,
                execute,
                subscribe,
            },
            {
                server: this.wsServer,
                path: '/graphql',
            },
        );

        const app = express();
        // @ts-ignore
        apolloServer.applyMiddleware({ app });
        app.use(express.static(Path.join(__dirname, 'interface')))
        app.get("/*", (req, res) => {
            res.sendFile(Path.join(__dirname, 'interface/index.html'))
        })
        this.server = app
    }

    run() {
        console.log("Starting server...")

        this.data_handler.startWatcher()

        this.wsServer.listen(3001, () => console.log(
            `Websocket Server is now running on http://localhost:${this.PORT + 1}`
        ));

        this.server.listen({port: this.PORT},() => {
            console.log(`Visualizer server ready at http://localhost:${this.PORT}`);
        });
    }
}
