import JSON5 from 'json5';
import autoBind from 'auto-bind';
import { defer, debounce } from 'lodash';

export default class JupyterComm {
  interfaceId: string;
  jcomm: InnerJupyterComm;
  resolvers: { [seqNum: number]: (value: unknown) => void };
  seqNumber: number;

  constructor(interfaceId) {
    autoBind(this);
    this.interfaceId = interfaceId;
    this.jcomm = new InnerJupyterComm('adatest_interface_target_'+this.interfaceId, this.handleResponse);
    this.resolvers = {};
    this.seqNumber = 0;
  }

  send(keys, data) {
    console.log("JUPYTERCOMM send", keys, data)
    const pendingData = this.addPendingData(keys, data);
    return this.sendPendingData(pendingData);
  }

  sendEvent(commEvent) {
    console.log("JUPYTERCOMM sendEvent", commEvent)
    let pendingData = {};
    for (const k of Object.keys(commEvent)) {
      this.addPendingData(k, commEvent[k], pendingData);
    }
    return this.sendPendingData(pendingData);
  }

  addPendingData(keys, data, pendingData={}) {
    console.log("JUPYTERCOMM addPendingData", keys, data, pendingData)
    if (!Array.isArray(keys)) keys = [keys];
    for (const k of keys) {
      pendingData[k] = data;
    }
    return pendingData;
  }

  handleResponse(data) {
    console.log("JUPYTERCOMM handleResponse", data)
    data = JSON5.parse(data["data"]) // data from Jupyter is wrapped so we get to do our own JSON encoding
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

  getSeqNumber() {
    return this.seqNumber++;
  }

  sendPendingData(pendingData: any) {
    console.log("JUPYTERCOMM sendPendingData", pendingData)

    const seqNumber = this.getSeqNumber();
    const promise = new Promise((resolve, reject) => {
      const timeout_ms = 60000;
      setTimeout(() => {
        reject(`timeout waiting for response to message#${seqNumber}`);
      }, timeout_ms);
      console.log(`sending message#${seqNumber}`, pendingData);
      pendingData["sequence_number"] = seqNumber;
      this.resolvers[seqNumber] = resolve;
      this.jcomm.send_data(pendingData);
    });
    return promise;
  }
}

class InnerJupyterComm {
  jcomm: any;
  callback: any;

  constructor(target_name, callback, mode="open") {
    this._fire_callback = this._fire_callback.bind(this);
    this._register = this._register.bind(this)

    this.jcomm = undefined;
    this.callback = callback;

    // https://jupyter-notebook.readthedocs.io/en/stable/comms.html
    if (mode === "register") {
      // @ts-ignore
      Jupyter.notebook.kernel.comm_manager.register_target(target_name, this._register);
    } else {
      // @ts-ignore
      this.jcomm = Jupyter.notebook.kernel.comm_manager.new_comm(target_name);
      this.jcomm.on_msg(this._fire_callback);
    }
  }

  send_data(data) {
    console.log("INNERJUPYTERCOMM send_data", data)
    if (this.jcomm !== undefined) {
      this.jcomm.send(data);
    } else {
      console.error("Jupyter comm module not yet loaded! So we can't send the message.")
    }
  }

  _register(jcomm, msg) {
    console.log("INNERJUPYTERCOMM _register", jcomm)
    this.jcomm = jcomm;
    this.jcomm.on_msg(this._fire_callback);
  }

  _fire_callback(msg) {
    console.log("INNERJUPYTERCOMM _fire_callback", msg)
    this.callback(msg.content.data)
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