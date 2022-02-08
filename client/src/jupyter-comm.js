

export default class JupyterComm {
  constructor(target_name, callback, mode="open") {
    this._fire_callback = this._fire_callback.bind(this);
    this._register = this._register.bind(this)

    this.jcomm = undefined;
    this.callback = callback;

    // https://jupyter-notebook.readthedocs.io/en/stable/comms.html
    if (mode === "register") {
      Jupyter.notebook.kernel.comm_manager.register_target(target_name, this._register);
    } else {
      this.jcomm = Jupyter.notebook.kernel.comm_manager.new_comm(target_name);
      this.jcomm.on_msg(this._fire_callback);
    }
  }

  send_data(data) {
    if (this.jcomm !== undefined) {
      this.jcomm.send(data);
    } else {
      console.error("Jupyter comm module not yet loaded! So we can't send the message.")
    }
  }

  _register(jcomm, msg) {
    this.jcomm = jcomm;
    this.jcomm.on_msg(this._fire_callback);
  }

  _fire_callback(msg) {
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