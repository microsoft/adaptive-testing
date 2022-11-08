import JSON5 from 'json5';
import autoBind from 'auto-bind';
import { defer, debounce } from 'lodash';
import { CommEvent } from './CommEvent';

export default class WebSocketComm {
  interfaceId: string;
  websocketServer: string;
  callbackMap: { [key: string]: (data: any) => void };
  // data to send to the server
  pendingData: {};
  // data received from the server
  pendingResponses: {};
  wcomm: WebSocket;
  reconnectDelay: number;
  seqNumber: number;

  constructor(interfaceId, websocketServer) {
    autoBind(this);
    this.interfaceId = interfaceId;
    this.websocketServer = websocketServer;
    this.callbackMap = {};
    this.pendingData = {};
    this.pendingResponses = {};
    this.reconnectDelay = 100;
    this.seqNumber = 0;
  }

  send(keys, data) {
    this.addPendingData(keys, data);
    return this.sendPendingData();
  }

  sendEvent(commEvent: CommEvent) {
    for (const k of Object.keys(commEvent)) {
      this.addPendingData(k, commEvent[k]);
    }
    return this.sendPendingData();
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

  addPendingData(keys, data) {
    console.log("addPendingData", keys, data);
    if (!Array.isArray(keys)) keys = [keys];
    for (const i in keys) {
      const k = keys[i];
      this.pendingData[k] = data;
    }
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
      this.pendingResponses[data.sequence_number] = data;
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

  sendPendingData() {
    const seqNumber = this.getSeqNumber();
    this.pendingData["sequence_number"] = seqNumber;
    console.log(`sending message#${seqNumber}`, this.pendingData);
    this.wcomm.send(JSON.stringify(this.pendingData));
    this.pendingData = {};
    return this.waitForResponse(seqNumber);
  }

  waitForResponse(seqNumber): Promise<any> {
    const timeout_ms = 60000;
    this.pendingResponses[seqNumber] = "pending";
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(`timeout waiting for response to message#${seqNumber}`);
      }, timeout_ms);
      const interval = setInterval(() => {
        if (this.pendingResponses[seqNumber] !== "pending") {
          clearTimeout(timeout);
          clearInterval(interval);
          const responseData = this.pendingResponses[seqNumber];
          this.pendingResponses[seqNumber] = undefined;
          resolve(responseData);
        }
      }, 100);
    });
  }

}