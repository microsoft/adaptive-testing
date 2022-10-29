import { changeTopic, redraw } from "./CommEvent";
import JupyterComm from "./jupyter-comm";
import { AppDispatch } from "./store";
import { refresh } from "./TestTreeSlice";
import { Comm } from "./types";
import WebSocketComm from "./web-socket-comm";

export function refreshBrowser(comm: Comm, dispatch: AppDispatch) {
  return comm.sendEvent(redraw()).then((data) => {
    if (data["status"] === "ok") {
      dispatch(refresh(data["data"]));
    } else {
      // TODO: handle error
    }
  });
}

export function goToTopic(topic: string, comm: Comm) {
  console.log("goToTopic", topic);
  // if (this.suggestionsTemplateRow) {
  //   this.suggestionsTemplateRow.setState({value2: null});
  // }
  return comm.sendEvent(changeTopic(stripSlash(topic).replaceAll(" ", "%20")));
}

export function useComm(env: string, interfaceId: any, websocket_server: string="") {
  console.log("pairs interfaceId", interfaceId)
  if (env === "jupyter") {
    return new JupyterComm(interfaceId);
  } else if (env === "web") {
    if (websocket_server !== "") {
      return new WebSocketComm(interfaceId, websocket_server);
    } else {
      console.error("websocket_server is not set");
      throw new Error("websocket_server is not set");
    }
  } else {
    console.error("Unknown environment:", env);
    throw new Error(`Unknown environment: ${env}`);
  }
}

export function stripPrefix(path: any, prefix: any) {
  if (path.startsWith(prefix)) {
    return path.slice(prefix.length);
  } else {
    return path;
  }
}

export function stripSlash(str) {
  return str.endsWith('/') ? str.slice(0, -1) : str;
}