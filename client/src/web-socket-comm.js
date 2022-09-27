import JSON5 from 'json5';
import autoBind from 'auto-bind';
import { defer, debounce } from 'lodash';

export default class WebSocketComm {
  constructor(interfaceId, websocketServer, onopen) {
    autoBind(this);
    this.interfaceId = interfaceId;
    this.websocketServer = websocketServer;
    this.callbackMap = {};
    // this.data = {};
    this.pendingData = {};
    this.pendingResponses = {};
    this.onopen = onopen;
    this.reconnectDelay = 100;
    this.seqNumber = 0;

    this.debouncedSendPendingData500 = debounce(this.sendPendingData, 500);
    this.debouncedSendPendingData1000 = debounce(this.sendPendingData, 1000);

    this.connect();
  }

  send(keys, data) {
    this.addPendingData(keys, data);
    this.sendPendingData();
  }

  sendEvent(commEvent) {
    for (const k of Object.keys(commEvent)) {
      this.addPendingData(k, commEvent[k]);
    }
    this.sendPendingData();
  }

  debouncedSendEvent500(commEvent) {
    for (const k of Object.keys(commEvent)) {
      this.addPendingData(k, commEvent[k]);
    }
    this.debouncedSendPendingData500();
  }

  debouncedSend500(keys, data) {
    this.addPendingData(keys, data);
    this.debouncedSendPendingData500();
  }

  debouncedSend1000(keys, data) {
    this.addPendingData(keys, data);
    this.debouncedSendPendingData1000();
  }

  addPendingData(keys, data) {
    // console.log("addPendingData", keys, data);
    if (!Array.isArray(keys)) keys = [keys];
    for (const i in keys) {
      const k = keys[i];
      this.pendingData[k] = data;
      // this.data[k] = Object.assign(this.data[k] || {}, data); // pretend it has already changed in our data cache
    }
  }

  connect() {
    let wsUri = (window.location.protocol=='https:' ? 'wss://' : 'ws://') + (this.websocketServer.startsWith("/") ? window.location.host : "") + this.websocketServer;
    this.wcomm = new WebSocket(wsUri);
    this.wcomm.onopen = this.onopen;
    this.wcomm.onmessage = this.handleResponse;
    this.wcomm.onerror = this.onError;
    this.wcomm.onclose = this.onClose;
  }

  handleResponse(e) {
    let data = JSON5.parse(e.data);
    const keys = Object.keys(data);
    if (keys.includes("sequence_number")) {
      console.log(`received message#${data.sequence_number}`, data);
    }
  }

  // updateData(e) {
  //   console.log("WEBSOCKET UPDATEDATA, received unexpected data", data)
    // for (const k in data) {
    //   // console.log("data[k]", data[k])
    //   this.data[k] = Object.assign(this.data[k] || {}, data[k]);
    //   if (k in this.callbackMap) {
    //     this.callbackMap[k](data[k]);
    //   }
    // }
  // }

  onError(e) {
    console.log("Websocket error", e);
  }

  onClose(e) {
    console.log('Socket is closed. Reconnect will be attempted...', e.reason);
    setTimeout(this.connect, this.reconnectDelay);
    this.reconnectDelay += 1000;
  }

  subscribe(key, callback) {
    console.log("WEBSOCKET SUBSCRIBE", key, callback);
    // this.callbackMap[key] = callback;
    // defer(_ => this.callbackMap[key](this.data[key]));
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

  waitForResponse(seqNumber) {
    const timeout_ms = 30000;
    this.pendingResponses[seqNumber] = "pending";
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(`timeout waiting for response to message#${seqNumber}`);
      }, timeout_ms);
      const interval = setInterval(() => {
        if (this.pendingResponses[seqNumber] !== "pending") {
          clearTimeout(timeout);
          clearInterval(interval);
          resolve(this.pendingResponses[seqNumber]);
        }
      }, 10);
    });
  }

}