import "./adatest.css";

import React from 'react';
import ReactDOM from 'react-dom';
import Browser from './browser'
import {
  useLocation,
  useNavigate,
  useParams,
  BrowserRouter,
  MemoryRouter
} from "react-router-dom";

// function withRouter(Component) {
//   function ComponentWithRouterProp(props) {
//     let location = useLocation();
//     let navigate = useNavigate();
//     let params = useParams();
//     return (
//       <Component
//         {...props}
//         router={{ location, navigate, params }}
//       />
//     );
//   }

//   return ComponentWithRouterProp;
// }

// const BrowserWithRouter = withRouter(Browser);

export default class AdaTest extends React.Component {

  constructor(props) {
    super(props);
    console.log("interfaceId", this.props.interfaceId)
    this.state = { enabled: true };
    window.adatest_root = this;
  }
  render() {

    // let location = useLocation();
    // let navigate = useNavigate();
    // let params = useParams();

    const Router = this.props.environment === "web" ? BrowserRouter : MemoryRouter;

    return (
      <div style={{maxWidth: "1200px", marginLeft: "auto", marginRight: "auto"}}>
        <div style={{paddingLeft: "0px", width: "100%", fontFamily: "Helvetica Neue, Helvetica, Arial, sans-serif", boxSizing: "border-box", fontSize: "13px", opacity: this.state.enabled ? 1 : 0.4}}>
          <Router>
            <Browser
              interfaceId={this.props.interfaceId} environment={this.props.environment}
              websocket_server={this.props.websocket_server} enabled={this.state.enabled}
              startingTopic={this.props.startingTopic} prefix={this.props.prefix}
              // router={{ location, navigate, params }}
            />
          </Router>
        </div>
      </div>
    );
  }
}

window.AdaTestReact = React
window.AdaTestReactDOM = ReactDOM
window.AdaTest = AdaTest

