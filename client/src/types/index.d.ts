import JupyterComm from "src/jupyter-comm";
import WebSocketComm from "src/web-socket-comm";

export {};

declare global {
  // Declare all the variables that we add to the global Window object
  // Otherwise, TypeScript will complain that these variables are not defined
  interface Window {
    adatest_root: any;
    AdaTestReact: any;
    AdaTestReactDOM: any;
    AdaTest: any;
    faTimes: any;
  }
}

export type Comm = JupyterComm | WebSocketComm;