import JSON5 from 'json5';
import autoBind from 'auto-bind';
import { defer, debounce } from 'lodash';
import { CommEvent } from './CommEvent';

export default class WebSocketComm {
  interfaceId: string;
  websocketServer: string;
  callbackMap: { [key: string]: (data: any) => void };
  // data to send to the server
  // map of seqNum to resolve function for the promise
  resolvers: { [seqNum: number]: (value: unknown) => void };
  wcomm: WebSocket;
  reconnectDelay: number;
  seqNumber: number;

  constructor(interfaceId, websocketServer) {
    autoBind(this);
    this.interfaceId = interfaceId;
    this.websocketServer = websocketServer;
    this.callbackMap = {};
    this.resolvers = {};
    this.reconnectDelay = 100;
    this.seqNumber = 0;
  }

  send(keys, data) {
    const pendingData = this.addPendingData(keys, data);
    return this.sendPendingData(pendingData);
  }

  sendEvent(commEvent: CommEvent) {
    let pendingData = {};
    for (const k of Object.keys(commEvent)) {
      this.addPendingData(k, commEvent[k], pendingData);
    }
    return this.sendPendingData(pendingData);
  }

  // debouncedSendEvent500(commEvent) {
  //   for (const k of Object.keys(commEvent)) {
  //     this.addPendingData(k, commEvent[k]);
  //   }
  //   this.debouncedSendPendingData500();
  // }

  // debouncedSend500(keys, data) {
  //   this.addPendingData(keys, data);
  //   this.debouncedSendPendingData500();
  // }

  // debouncedSend1000(keys, data) {
  //   this.addPendingData(keys, data);
  //   this.debouncedSendPendingData1000();
  // }

  addPendingData(keys, data, pendingData={}) {
    console.log("addPendingData", keys, data);
    if (!Array.isArray(keys)) keys = [keys];
    for (const k of keys) {
      pendingData[k] = data;
    }
    return pendingData;
  }

  connect(onOpen: any) {
    let wsUri = (window.location.protocol=='https:' ? 'wss://' : 'ws://') + (this.websocketServer.startsWith("/") ? window.location.host : "") + this.websocketServer;
    this.wcomm = new WebSocket(wsUri);
    this.wcomm.onopen = onOpen;
    this.wcomm.onmessage = this.handleResponse;
    this.wcomm.onerror = this.onError;
    this.wcomm.onclose = this.onClose;
  }

  handleResponse(e) {
    let data = JSON5.parse(e.data);
    const keys = Object.keys(data);
    if (keys.includes("sequence_number")) {
      console.log(`received message#${data.sequence_number}`, data);
      if (data.sequence_number in this.resolvers) {
        this.resolvers[data.sequence_number](data);
        delete this.resolvers[data.sequence_number];
      } else {
        console.log(`no resolver for message#${data.sequence_number}`);
      }
    }
  }

  onError(e) {
    console.log("Websocket error", e);
  }

  onClose(e) {
    console.log('Socket is closed. Reconnect will be attempted...', e.reason);
    setTimeout(this.connect, this.reconnectDelay);
    this.reconnectDelay += 1000;
  }

  getSeqNumber() {
    return this.seqNumber++;
  }

  sendPendingData(pendingData: any): Promise<any> {
    const seqNumber = this.getSeqNumber();
    const promise = new Promise((resolve, reject) => {
      const timeout_ms = 60000;
      setTimeout(() => {
        reject(`timeout waiting for response to message#${seqNumber}`);
      }, timeout_ms);
      console.log(`sending message#${seqNumber}`, pendingData);
      pendingData["sequence_number"] = seqNumber;
      this.resolvers[seqNumber] = resolve;
      this.wcomm.send(JSON.stringify(pendingData));
    });
    return promise;
  }

}