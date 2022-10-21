import "./adatest.css";

import React from 'react';
import ReactDOM from 'react-dom';
import { withRouter } from 'react-router-dom';
import { BrowserRouter } from "react-router-dom";
import { MemoryRouter } from 'react-router';
import Browser from './browser'
import { store } from './store'
import { Provider } from 'react-redux'

const BrowserWithRouter = withRouter(Browser);


interface AdaTestProps {
  interfaceId: string;
  environment: string;
  websocket_server: string;
  startingTopic: string;
  prefix: string;
}

interface AdaTestState {
  enabled: boolean;
}

export default class AdaTest extends React.Component<AdaTestProps, AdaTestState> {

  constructor(props) {
    super(props);
    console.log("interfaceId", this.props.interfaceId)
    this.state = { enabled: true };
    window.adatest_root = this;
  }

  render() {
    const Router = this.props.environment === "web" ? BrowserRouter : MemoryRouter;

    return (
      <Provider store={store}>
        <div style={{maxWidth: "1000px", marginLeft: "auto", marginRight: "auto"}}>
          <div style={{paddingLeft: "0px", width: "100%", fontFamily: "Helvetica Neue, Helvetica, Arial, sans-serif", boxSizing: "border-box", fontSize: "13px", opacity: this.state.enabled ? 1 : 0.4}}>
            { /* @ts-ignore: JSX element type 'Router' does not have any construct or call signatures */ }
            <Router>
              <BrowserWithRouter
                interfaceId={this.props.interfaceId} environment={this.props.environment}
                websocket_server={this.props.websocket_server} enabled={this.state.enabled}
                startingTopic={this.props.startingTopic} prefix={this.props.prefix}
              />
            </Router>
          </div>
        </div>
      </Provider>
    );
  }
}

window.AdaTestReact = React
window.AdaTestReactDOM = ReactDOM
window.AdaTest = AdaTest

