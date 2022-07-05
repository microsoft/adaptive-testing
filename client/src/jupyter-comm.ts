import JSON5 from 'json5';
import autoBind from 'auto-bind';
import { defer, debounce } from 'lodash';
import { CommData } from './CommData';
import InnerJupyterComm from "./inner-jupyter-comm";

export default class JupyterComm {
  interfaceId: any;
  callbackMap: {};
  data: {};
  pendingData: {};
  jcomm: InnerJupyterComm;
  debouncedSendPendingData500: any;
  debouncedSendPendingData1000: any;

  constructor(interfaceId, onopen) {
    autoBind(this);
    this.interfaceId = interfaceId;
    this.callbackMap = {};
    this.data = {};
    this.pendingData = {};
    this.jcomm = new InnerJupyterComm('adatest_interface_target_'+this.interfaceId, this.updateData);

    this.debouncedSendPendingData500 = debounce(this.sendPendingData, 500);
    this.debouncedSendPendingData1000 = debounce(this.sendPendingData, 1000);
    if (onopen) {
      defer(onopen);
    }
  }

  send(keys, data) {
    this.addPendingData(keys, data);
    this.sendPendingData();
  }

  sendCommData(commData: CommData) {
    this.send(commData.event_id, commData.data);
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
    for (const i in keys) this.pendingData[keys[i]] = data;
  }

  updateData(data) {
    data = JSON5.parse(data["data"]) // data from Jupyter is wrapped so we get to do our own JSON encoding
    console.log("updateData", data)

    // save the data locally
    for (const k in data) {
      this.data[k] = data[k];
    }

    // call all the registered callbacks
    for (const k in data) {
      if (k in this.callbackMap) {
        this.callbackMap[k](this.data[k]);
      }
    }
  }

  subscribe(key, callback) {
    this.callbackMap[key] = callback;
    defer(_ => this.callbackMap[key](this.data[key]));
  }

  sendPendingData() {
    console.log("sending", this.pendingData);
    this.jcomm.send_data(this.pendingData);
    this.pendingData = {};
  }
}


// const comm = JupyterComm();

// // Jupyter.notebook.kernel.comm_manager.register_target('gadfly_comm_target',
// //   function(jcomm, msg) {
// //     // comm is the frontend comm instance
// //     // msg is the comm_open message, which can carry data

// //     comm.jcomm = jcomm

// //     // Register handlers for later messages:
// //     inner_comm.on_msg(function(msg) { console.log("MSGG", msg); });
// //     inner_comm.on_close(function(msg) { console.log("MSGdG", msg); });
// //     comm.send({'foo': 0});
// //   }
// // );

// export default comm;