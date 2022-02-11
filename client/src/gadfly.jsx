import "./gadfly.css";

import './jupyter-comm'
import React from 'react';
// import { useLocation } from 'react-router-dom'
import {withRouter} from 'react-router-dom';
import {
  BrowserRouter,
  Switch,
  Route,
  Link,
  useParams
} from "react-router-dom";
import { MemoryRouter } from 'react-router';
import JSON5 from 'json5';
import ReactDom from 'react-dom';
import autoBind from 'auto-bind';
import sanitizeHtml from 'sanitize-html';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faFolderPlus, faCheck, faTimes, faFolder, faChevronDown, faRedo, faSearch, faFilter } from '@fortawesome/free-solid-svg-icons'
import JupyterComm from './jupyter-comm'
//import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
//import { makeStyles } from '@material-ui/core/styles';
//import { TreeView, TreeItem } from '@material-ui/lab';
//import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
//import ChevronRightIcon from '@material-ui/icons/ChevronRight';
// import { useDrag } from 'react-dnd'
import { defer, debounce, partial, clone, get } from 'lodash';


//const red_blue_100 = ["rgb(0.0,138.56128015770724,250.76166088685727,255.0)","rgb(0.0,137.4991163711455,250.4914687565557,255.0)","rgb(0.0,135.89085862817228,250.03922790292606,255.0)","rgb(0.0,134.80461722068503,249.69422979450337,255.0)","rgb(0.0,133.15912944070257,249.12764143629818,255.0)","rgb(0.0,132.04779673175938,248.70683279399356,255.0)","rgb(0.0,130.3634759186023,248.02444138814778,255.0)","rgb(0.0,128.65565323564863,247.27367576741693,255.0)","rgb(0.0,127.50110843874282,246.72753679433836,255.0)","rgb(0.0,125.75168029462561,245.85912208200173,255.0)","rgb(0.0,124.56903652403216,245.23521693285122,255.0)","rgb(0.0,122.77608206265468,244.24829742509777,255.0)","rgb(0.0,120.95599474876376,243.18984596934288,255.0)","rgb(0.0,119.72546791225868,242.44012441018438,255.0)","rgb(0.0,117.8591797836317,241.26416165478395,255.0)","rgb(0.0,116.59613419778282,240.4339311004283,255.0)","rgb(0.0,114.68050681628627,239.1383865197414,255.0)","rgb(0.0,113.3839993210777,238.2294590645131,255.0)","rgb(0.0,111.41634894068424,236.81437229328455,255.0)","rgb(24.588663906345325,109.41632410184977,235.32817682974928,255.0)","rgb(35.44735081278475,108.06183480151708,234.29254074792976,255.0)","rgb(48.051717444228224,106.00540836596966,232.68863110680456,255.0)","rgb(54.58382033054716,104.61144706132748,231.57374376885096,255.0)","rgb(63.262053865061056,102.49432802815242,229.85160463157354,255.0)","rgb(70.76957320138267,100.33771880021955,228.05703474246997,255.0)","rgb(75.30021073284686,98.87675486589102,226.81988465865538,255.0)","rgb(81.62949947778507,96.6540104818813,224.91180462302458,255.0)","rgb(85.51695933372014,95.1445958160281,223.59546445853985,255.0)","rgb(91.05223198159915,92.84880956154888,221.57285170783555,255.0)","rgb(94.50489166579793,91.28823726624896,220.18077895318467,255.0)","rgb(99.4583420952925,88.91232187683374,218.04471819705824,255.0)","rgb(104.12179041241362,86.48354864452708,215.84022539728738,255.0)","rgb(107.08110644040013,84.83182437240492,214.33114916353256,255.0)","rgb(111.35380618049061,82.31139742285828,212.01929775886506,255.0)","rgb(114.06578608562516,80.5910630456271,210.4357328916578,255.0)","rgb(118.00136409704606,77.96485213886614,208.016034988954,255.0)","rgb(121.74766173376136,75.26429120233696,205.5295010169315,255.0)","rgb(124.15338187699085,73.42230736362347,203.83759830570943,255.0)","rgb(127.6354972896637,70.59265795441205,201.2503012002821,255.0)","rgb(129.8659364639127,68.65218238542003,199.48871459726595,255.0)","rgb(133.10804101178616,65.66976521198737,196.80305057967465,255.0)","rgb(136.20117312715468,62.56838858187191,194.0535664536312,255.0)","rgb(138.1973647491704,60.43984691426682,192.1917348903892,255.0)","rgb(141.0812080782349,57.12602871008347,189.34981967460325,255.0)","rgb(142.93497217316724,54.83018307067161,187.42427995933073,255.0)","rgb(145.62183484149273,51.24379487478906,184.4934665197036,255.0)","rgb(147.34120668891256,48.72446502544426,182.50692554291248,255.0)","rgb(149.83882553726863,44.76410333626281,179.49010263854294,255.0)","rgb(152.21629692400205,40.46878966316789,176.41890349035555,255.0)","rgb(153.74537831964477,37.405899634336166,174.3463167526223,255.0)","rgb(156.8946867035305,33.63427487751921,172.10689873421705,255.0)","rgb(159.57052897508228,31.76356558772026,171.1904317530127,255.0)","rgb(163.51569605559672,28.753894553429824,169.77296032725718,255.0)","rgb(167.36700142841255,25.389234485648817,168.29649352383,255.0)","rgb(169.8939277417104,22.949167366394107,167.28653276271265,255.0)","rgb(173.60751889004476,18.751898233902914,165.72304707948427,255.0)","rgb(176.04037452587522,15.53349266017855,164.65367757038166,255.0)","rgb(179.62227517307235,9.586956240670718,163.00736797488793,255.0)","rgb(181.96398047712216,5.039485831605471,161.88104419124014,255.0)","rgb(185.4175817560734,0.0,160.1552802934102,255.0)","rgb(188.7866522134373,0.0,158.37821514882566,255.0)","rgb(190.99545744496794,0.0,157.1712500750526,255.0)","rgb(194.23939479374397,0.0,155.32022934668817,255.0)","rgb(196.3613372562259,0.0,154.06279982720955,255.0)","rgb(199.48227089890267,0.0,152.14184153517203,255.0)","rgb(202.52395205893734,0.0,150.17747004313574,255.0)","rgb(204.51720731357923,0.0,148.84941784048203,255.0)","rgb(207.43415034062437,0.0,146.8195255831636,255.0)","rgb(209.34227187082465,0.0,145.447819126977,255.0)","rgb(212.1382144344341,0.0,143.35790157156467,255.0)","rgb(214.85836244585704,0.0,141.23222265886395,255.0)","rgb(216.63670670950123,0.0,139.79905843731933,255.0)","rgb(219.23301240468484,0.0,137.6179798879229,255.0)","rgb(220.92924595032096,0.0,136.14914842114652,255.0)","rgb(223.4023955123896,0.0,133.91677798534914,255.0)","rgb(225.0139515716111,0.0,132.41381956928373,255.0)","rgb(227.36646430400836,0.0,130.13484833504174,255.0)","rgb(229.64256707387395,0.0,127.82818117110327,255.0)","rgb(231.1250651865189,0.0,126.27823065434256,255.0)","rgb(233.27714151990378,0.0,123.9294477401314,255.0)","rgb(234.676703289514,0.0,122.35230960074493,255.0)","rgb(236.70590614401792,0.0,119.96514396719735,255.0)","rgb(238.66106706741957,0.0,117.55637241505875,255.0)","rgb(239.92865207922344,0.0,115.94046512913654,255.0)","rgb(241.75947067582808,0.0,113.49767303291209,255.0)","rgb(242.94521123855867,0.0,111.86012465750336,255.0)","rgb(244.6516179552062,0.0,109.38587603007474,255.0)","rgb(245.75354715682175,0.0,107.72779565577645,255.0)","rgb(247.33745729848604,0.0,105.22467948947163,255.0)","rgb(248.84659658395643,0.0,102.70475812439781,255.0)","rgb(249.8170331745849,0.0,101.01697337568197,255.0)","rgb(251.20184136638093,0.0,98.47001746051721,255.0)","rgb(252.09049986637743,0.0,96.76466947151692,255.0)","rgb(253.35113135488535,0.0,94.19124565826175,255.0)","rgb(254.475785780405,0.0,91.60224197581371,255.0)","rgb(255.0,0.0,89.86831280400678,255.0)","rgb(255.0,0.0,87.25188871633031,255.0)","rgb(255.0,0.0,85.49944591423251,255.0)","rgb(255.0,0.0,82.8535189165512,255.0)","rgb(255.0,0.0,81.08083606031792,255.0)"]
const red_blue_100 = ["rgb(0.0, 199.0, 100.0)", "rgb(7.68, 199.88, 96.12)", "rgb(15.36, 200.76, 92.24)", "rgb(23.04, 201.64, 88.36)", "rgb(30.72, 202.52, 84.48)", "rgb(38.4, 203.4, 80.6)", "rgb(46.08, 204.28, 76.72)", "rgb(53.76, 205.16, 72.84)", "rgb(61.44, 206.04, 68.96)", "rgb(69.12, 206.92, 65.08)", "rgb(76.8, 207.8, 61.2)", "rgb(84.48, 208.68, 57.32)", "rgb(92.16, 209.56, 53.44)", "rgb(99.84, 210.44, 49.56)", "rgb(107.52, 211.32, 45.68)", "rgb(115.2, 212.2, 41.8)", "rgb(122.88, 213.08, 37.92)", "rgb(130.56, 213.96, 34.04)", "rgb(138.24, 214.84, 30.16)", "rgb(145.92, 215.72, 26.28)", "rgb(153.6, 216.6, 22.4)", "rgb(161.28, 217.48, 18.52)", "rgb(168.96, 218.36, 14.64)", "rgb(176.64, 219.24, 10.76)", "rgb(184.32, 220.12, 6.88)", "rgb(192.0, 221.0, 3.0)", "rgb(194.52, 220.12, 6.44)", "rgb(197.04, 219.24, 9.88)", "rgb(199.56, 218.36, 13.32)", "rgb(202.08, 217.48, 16.76)", "rgb(204.6, 216.6, 20.2)", "rgb(207.12, 215.72, 23.64)", "rgb(209.64, 214.84, 27.08)", "rgb(212.16, 213.96, 30.52)", "rgb(214.68, 213.08, 33.96)", "rgb(217.2, 212.2, 37.4)", "rgb(219.72, 211.32, 40.84)", "rgb(222.24, 210.44, 44.28)", "rgb(224.76, 209.56, 47.72)", "rgb(227.28, 208.68, 51.16)", "rgb(229.8, 207.8, 54.6)", "rgb(232.32, 206.92, 58.04)", "rgb(234.84, 206.04, 61.48)", "rgb(237.36, 205.16, 64.92)", "rgb(239.88, 204.28, 68.36)", "rgb(242.4, 203.4, 71.8)", "rgb(244.92, 202.52, 75.24)", "rgb(247.44, 201.64, 78.68)", "rgb(249.96, 200.76, 82.12)", "rgb(252.48, 199.88, 85.56)", "rgb(255.0, 199.0, 89.0)", "rgb(255.0, 197.36, 87.08)", "rgb(255.0, 195.72, 85.16)", "rgb(255.0, 194.08, 83.24)", "rgb(255.0, 192.44, 81.32)", "rgb(255.0, 190.8, 79.4)", "rgb(255.0, 189.16, 77.48)", "rgb(255.0, 187.52, 75.56)", "rgb(255.0, 185.88, 73.64)", "rgb(255.0, 184.24, 71.72)", "rgb(255.0, 182.6, 69.8)", "rgb(255.0, 180.96, 67.88)", "rgb(255.0, 179.32, 65.96)", "rgb(255.0, 177.68, 64.04)", "rgb(255.0, 176.04, 62.12)", "rgb(255.0, 174.4, 60.2)", "rgb(255.0, 172.76, 58.28)", "rgb(255.0, 171.12, 56.36)", "rgb(255.0, 169.48, 54.44)", "rgb(255.0, 167.84, 52.52)", "rgb(255.0, 166.2, 50.6)", "rgb(255.0, 164.56, 48.68)", "rgb(255.0, 162.92, 46.76)", "rgb(255.0, 161.28, 44.84)", "rgb(255.0, 159.64, 42.92)", "rgb(255.0, 158.0, 41.0)", "rgb(255.0, 153.25, 39.583)", "rgb(255.0, 148.5, 38.167)", "rgb(255.0, 143.75, 36.75)", "rgb(255.0, 139.0, 35.333)", "rgb(255.0, 134.25, 33.917)", "rgb(255.0, 129.5, 32.5)", "rgb(255.0, 124.75, 31.083)", "rgb(255.0, 120.0, 29.667)", "rgb(255.0, 115.25, 28.25)", "rgb(255.0, 110.5, 26.833)", "rgb(255.0, 105.75, 25.417)", "rgb(255.0, 101.0, 24.0)", "rgb(255.0, 96.25, 22.583)", "rgb(255.0, 91.5, 21.167)", "rgb(255.0, 86.75, 19.75)", "rgb(255.0, 82.0, 18.333)", "rgb(255.0, 77.25, 16.917)", "rgb(255.0, 72.5, 15.5)", "rgb(255.0, 67.75, 14.083)", "rgb(255.0, 63.0, 12.667)", "rgb(255.0, 58.25, 11.25)", "rgb(255.0, 53.5, 9.833)", "rgb(255.0, 48.75, 8.417)", "rgb(255.0, 44.0, 7.0)"]

function red_blue_color(value, min, max) {
  return red_blue_100[Math.floor(99.9999 * (value - min)/(max - min))]
}

const score_min = -1;
const score_max = 1;
function scale_score(score) {
  return Math.max(Math.min(score, score_max), score_min) ///(score_max - score_min)
}

function findParentWithClass(el, className) {
  const orig_el = el;
  while (el && !el.className.includes(className)) {
    el = el.parentElement;
  }
  return el ? el : orig_el;
}

class Clock extends React.Component {
  constructor(props) {
    super(props);
    autoBind(this);
    this.state = {date: new Date()};
    this.startTime = new Date();
    window.last_clock = this; // used for things like: window.last_clock.startTime.setSeconds(window.last_clock.startTime.getSeconds() - 60 * 4)
  }

  componentDidMount() {
    this.timerID = setInterval(
      () => this.tick(),
      1000
    );
  }

  componentWillUnmount() {
    clearInterval(this.timerID);
  }

  tick() {
    let now = new Date()
    if (now - this.startTime > this.props.duration*1000) {
      now = this.startTime;
      if (this.props.onFinish) {
        this.props.onFinish();
      }
      this.startTime = new Date();
    }
    this.setState({
      date: now
    });
    
  }

  clickClock(e) {
    console.log("clickClock", e, this)
    if (e.shiftKey) {
      this.startTime.setSeconds(this.startTime.getSeconds() - 60);
    } else if (e.metaKey) {
      this.startTime.setSeconds(this.startTime.getSeconds() + 60);
    }
  }

  render() {
    let sec = Math.round(this.props.duration - (this.state.date - this.startTime) / 1000);
    const min = Math.floor(sec / 60);
    sec = sec % 60;
    return (
      <div style={{textAlign: "center", fontSize: "16px"}}>
        <br />
        Time left for this task:&nbsp;<span onClick={this.clickClock} style={{fontSize: "16px", borderRadius: "6px", background: "#333333", color: "#ffffff", padding: "10px", marginTop: "10px"}}>{min}:{String(sec).padStart(2, '0')}</span>
      </div>
    );
  }
}

class GadflyJupyterComm {
  constructor(interfaceId, onopen) {
    autoBind(this);
    this.interfaceId = interfaceId;
    this.callbackMap = {};
    this.data = {};
    this.pendingData = {};
    this.jcomm = new JupyterComm('gadfly_interface_target_'+this.interfaceId, this.updateData);

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

  debouncedSend500(keys, data) {
    this.addPendingData(keys, data);
    this.debouncedSendPendingData500();
  }

  debouncedSend500(keys, data) {
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

class GadflyWebSocketComm {
  constructor(interfaceId, websocketServer, onopen) {
    autoBind(this);
    this.interfaceId = interfaceId;
    this.websocketServer = websocketServer;
    this.callbackMap = {};
    this.data = {};
    this.pendingData = {};
    this.onopen = onopen;
    this.reconnectDelay = 100;

    this.debouncedSendPendingData500 = debounce(this.sendPendingData, 500);
    this.debouncedSendPendingData1000 = debounce(this.sendPendingData, 1000);

    this.connect();
  }

  send(keys, data) {
    this.addPendingData(keys, data);
    this.sendPendingData();
  }

  debouncedSend500(keys, data) {
    this.addPendingData(keys, data);
    this.debouncedSendPendingData500();
  }

  debouncedSend500(keys, data) {
    this.addPendingData(keys, data);
    this.debouncedSendPendingData1000();
  }

  addPendingData(keys, data) {
    // console.log("addPendingData", keys, data);
    if (!Array.isArray(keys)) keys = [keys];
    for (const i in keys) this.pendingData[keys[i]] = data;
  }

  connect() {
    let wsUri = (window.location.protocol=='https:' ? 'wss://' : 'ws://') + (this.websocketServer.startsWith("/") ? window.location.host : "") + this.websocketServer;
    this.wcomm = new WebSocket(wsUri);
    this.wcomm.onopen = this.onopen;
    this.wcomm.onmessage = this.updateData;
    this.wcomm.onerror = this.onError;
    this.wcomm.onclose = this.onClose;
  }

  updateData(e) {
    console.log("updateData", e)
    let data = JSON5.parse(e.data);
    console.log("updateData", data)
    for (const k in data) {
      this.data[k] = data[k];
      if (k in this.callbackMap) {
        this.callbackMap[k](this.data[k]);
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

  subscribe(key, callback) {
    this.callbackMap[key] = callback;
    defer(_ => this.callbackMap[key](this.data[key]));
  }

  sendPendingData() {
    console.log("sending", this.pendingData);
    this.wcomm.send(JSON.stringify(this.pendingData));
    this.pendingData = {};
  }
}

function setCaret(el, pos) {
  var range = document.createRange();
  var sel = window.getSelection();
  
  range.setStart(el, pos)
  range.collapse(true)
  
  sel.removeAllRanges()
  sel.addRange(range)
}
document.setCaret = setCaret;

function selectElement(element){
  var doc = document;
  console.log(this, element);
  if (doc.body.createTextRange) {
      var range = document.body.createTextRange();
      range.moveToElementText(element);
      range.select();
  } else if (window.getSelection) {
      var selection = window.getSelection();        
      var range = document.createRange();
      range.selectNodeContents(element);
      selection.removeAllRanges();
      selection.addRange(range);
  }
}

class ContextMenu extends React.Component {
  static defaultProps = {
    top: 0,
    left: 0,
    open: false,
    onClick: () => undefined,
    rows: [
      "test"
    ]
  };

  constructor(props) {
    super(props);
    autoBind(this);
  }

  render() {
    return <div style={{display: this.props.open ? "inline-block" : "none"}}>
      <div style={{position: "fixed", top: 0, bottom: 0, left: 0, right: 0}} onClick={this.handleBackgroundClick}></div>
      <div style={{position: "absolute", top: this.props.top, left: this.props.left, background: "#ffffff", padding: "4px", boxShadow: "0px 2px 5px #666666"}}>
        {this.props.rows && this.props.rows.map(row => {
          return <div onClick={e => this.handleRowClick(row, e)} className="gadfly-hover-gray">{row}</div>
        })}
      </div>
    </div>
  }

  handleBackgroundClick(e) {
    e.preventDefault();
    this.props.onClose();
  }

  handleRowClick(row, e) {
    e.preventDefault();
    this.props.onClick(row);
  }
}

class ContentEditable extends React.Component {
  static defaultProps = {
    editable: true,
    defaultText: ""
  };

  constructor(props) {
    super(props);
    autoBind(this);
    this.lastText = null;

    this.divRef = {};
    window["cedit_"+this.props.id] = this;
  }

  render() {
    //console.log("this.props.text", this.props.text)
    const emptyContent = this.props.text == undefined || this.props.text.length === 0;
    this.lastEditable = this.props.editable;
    if (this.lastText === null) this.lastText = this.props.text;
    return <div
      ref={(el) => this.divRef = el}
      id={this.props.id}
      style={{opacity: emptyContent ? "0.3" : "1", display: "inline", overflowWrap: "anywhere"}}
      onFocus={this.onFocus}
      onInput={this.handleInput}
      onKeyPress={this.handleKeyPress}
      onKeyDown={this.handleKeyDown}
      onBlur={this.onBlur}
      onDoubleClick={this.handleDoubleClick}
      onDragStart={this.stopDrag}
      onClick={this.onClick}
      contentEditable={this.props.editable}
      className="gadfly-editable"
      dangerouslySetInnerHTML={{__html: sanitizeHtml(emptyContent ? this.props.defaultText : this.props.text)}}
      tabIndex="0"
    ></div>
  }

  stopDrag(e) {
    console.log("stopDrag")
    e.preventDefault();
    return false;
  }

  handleDoubleClick(e) {
    const range = getMouseEventCaretRange(e);
    console.log("handleDoubleClick", range, e)
  }

  focus() {

    // we blur without triggering an action so that we can refocus
    // this is important to get the cursor to come back sometimes
    this.skipBlurAction = true;
    this.divRef.blur();
    this.skipBlurAction = false;
    
    this.divRef.focus();
  }

  blur() {
    this.divRef.blur();
  }

  onFocus(e) {
    console.log("onFocus in ContentEditable", this.props.text);

    // if (!this.props.editing) return;
    
    if (this.props.text !== this.props.defaultText && this.divRef.innerText === this.props.defaultText) {
      e.preventDefault();
      e.stopPropagation();
      this.divRef.innerText = "";
      if (this.props.onClick) this.props.onClick(e); // why we need this is crazy to me, seems like setting inner text kills the click event
      // defer(() => this.focus());
      console.log("clear!!", this.props.editable)
      defer(() => this.focus());
    }
  }

  onClick(e) {
    // console.log("onClick in ContentEditable", this.props.onClick)
    if (this.props.onClick) {
      e.preventDefault();
      e.stopPropagation();
      this.props.onClick(e);
    }
    e.stopPropagation();
  }

  getValue() {
    const text = this.divRef.innerText;
    if (text === this.props.defaultText) return "";
    else return text;
  }

  shouldComponentUpdate(nextProps) {
    return nextProps.text !== this.divRef.innerText && (nextProps.text != "" || this.divRef.innerText != this.props.defaultText) || nextProps.editable != this.lastEditable;
  }

  componentDidUpdate() {
    this.componentDidUpdateOrMount(false);
  }

  componentDidMount() {
    this.componentDidUpdateOrMount(true);
  }
  
  componentDidUpdateOrMount(mount) {
    // console.log("ContentEditable componentDidUpdateOrMount", mount, this.props.text, this.props.editable);
    if (this.props.text !== this.divRef.innerText) {
      if (this.props.text !== undefined && this.props.text !== null && this.props.text.length > 0) {
        this.divRef.innerText = this.props.text;
      } else {
        if (mount) this.divRef.innerText = this.props.defaultText;
      }
    }
    if (this.props.text && (this.props.text.startsWith("New topic") || this.props.text === "New test") && this.props.editable) { // hacky but works for now
      // console.log("HACK!", this.props.text)
      this.divRef.focus();
      selectElement(this.divRef);
      // document.execCommand('selectAll', false, null);
    }
  }
      
  handleInput(e, finishing) {
    console.log("handleInput", finishing, this.divRef.innerText)
    const text = this.divRef.innerText;
    if (this.props.onInput && text !== this.lastText) {
      this.props.onInput(text);
      this.lastText = text;
    }

    if (finishing && this.props.onFinish) {
      this.props.onFinish(text);
    }

    if (text === this.props.defaultText) this.divRef.style.opacity = 0.3;
    else this.divRef.style.opacity = 1.0;
  }

  onBlur(e) {
    console.log("onBlur in ContentEditable", this.divRef.innerText, this.skipBlurAction)
    if (this.skipBlurAction) return;
    // if (this.divRef.innerText.length === this.props.defaultText) {
    //   this.divRef.innerText = "";
    // }
    this.handleInput(e, true);
    if (this.divRef.innerText.length === 0) {
      this.divRef.innerText = this.props.defaultText;
      this.divRef.style.opacity = 0.3;
    }
  }

  handleKeyPress(e) {

    console.log("handleKeyPress", e.charCode)
    e.stopPropagation();
    if (e.charCode == 13) {
      e.preventDefault();

      this.handleInput(e, true);
    }
  }

  handleKeyDown(e) {
    console.log("handleKeyDown", e.charCode)
    // only let the enter/return key go through
    if (e.charCode != 13) e.stopPropagation();
  }
}

// https://stackoverflow.com/questions/45408920/plain-javascript-scrollintoview-inside-div
function scrollParentToChild(parent, child) {
  console.log("scrollParentToChild", parent, child)
  // Where is the parent on page
  var parentRect = parent.getBoundingClientRect();
  let parentScrolls = (parent.scrollHeight - parent.clientHeight) > 0;

  // What can you see?
  var parentViewableArea = {
    height: parent.clientHeight,
    width: parent.clientWidth
  };

  const margin = 50;

  // Where is the child
  var childRect = child.getBoundingClientRect();
  // Is the child viewable?
  if (parentScrolls) {
    var isViewable = (childRect.top > margin) && (childRect.bottom + margin <= parentRect.top + parentViewableArea.height);
  } else {
    var isViewable = (childRect.top > margin) && (childRect.bottom + margin <= parentViewableArea.height);
  }
  

  // if you can't see the child try to scroll parent
  if (!isViewable) {
        // Should we scroll using top or bottom? Find the smaller ABS adjustment
        if (parentScrolls) {
          var scrollTop = childRect.top - parentRect.top;
          var scrollBot = childRect.bottom - parentViewableArea.height - parentRect.top;
        } else {
          var scrollTop = childRect.top;
          var scrollBot = childRect.bottom - parentViewableArea.height;
        }
        if (Math.abs(scrollTop) < Math.abs(scrollBot)) {
            // we're near the top of the list
            parent.scrollTop += scrollTop - margin;
        } else {
            // we're near the bottom of the list
            parent.scrollTop += scrollBot + margin;
        }
  }

}

class Row extends React.Component {
  constructor(props) {
    super(props);
    autoBind(this);

    if (this.props.value1Default === undefined) {
      this.props.value1Default = "New value";
    }
    if (this.props.value2Default === undefined) {
      this.props.value2Default = "New value";
    }

    this.state = {
      editing: false,
      value1: null,
      comparator: null,
      value2: null,
      topic_name: null,
      topic: null,
      scores: null,
      dragging: false,
      dropHighlighted: 0,
      hovering: false,
      plusHovering: false
    };

    this.dataLoadActions = [];

    this.props.comm.subscribe(this.props.id, this.dataLoaded);

    window["row_"+this.props.id] = this;
  }

  dataLoaded(state) {
    if (state == undefined) return;

    if (this.dataLoadActions.length > 0) {
      for (let i = 0; i < this.dataLoadActions.length; i++) {
        this.dataLoadActions[i]();
      }
      this.dataLoadActions = [];
    }
    console.log("state.topic_name", state.topic_name)
    // we automatically start editing topics that are selected and have an imputed name
    if (state.topic_name && (state.topic_name.startsWith("New topic") || state.value1 === "New test") && this.props.soleSelected) {
      state["editing"] = true;
      console.log("setting editing state to true!")
    }
    
    this.setState(state);
  }

  componentWillUpdate(nextProps, nextState) {

    // if we are becoming to sole selected item then we should scroll to be viewable after rendering
    if (!this.props.soleSelected && nextProps.soleSelected) {
      this.scrollToView = true;
    }

    // we need to force a relayout if the comparator changed since that imapact global alignments
    if (this.state.comparator !== nextState.comparator) {
      if (this.props.forceRelayout) this.props.forceRelayout();
    }
  }

  componentDidUpdate() {
    this.componentDidUpdateOrMount(false);
  }

  componentDidMount() {
    this.componentDidUpdateOrMount(true);
  }
  
  componentDidUpdateOrMount(mount) {
    // see if we should scroll to make ourselves visible
    if (this.scrollToView) {
      console.log("scrollingtoView!");
      if (this.divRef) {
        this.divRef.focus();
        scrollParentToChild(this.props.scrollParent, this.divRef);
        this.scrollToView = false;
      }
    }
  }

  render() {
    const main_score = this.props.scoreColumns ? this.props.scoreColumns[0] : undefined;

    // apply the value1/value2/topic filters
    let value1_outputs_str = "";
    let value2_outputs_str = "";
    if (this.state.topic_name === null) {
      if (this.props.value1Filter && this.state.value1 !== "") {
        const re = RegExp(this.props.value1Filter);
        if (!re.test(this.state.value1)) return null;
      }
      if (this.props.comparatorFilter) {
        const re = RegExp(this.props.comparatorFilter);
        if (!re.test(this.state.comparator)) {
          if (this.state.value1 === "") { // we are the blank suggestion
            for (const c of ["should not be", "should be", "should be the same as for", "should be invertable."]) {
              if (re.test(c)) {
                this.setState({comparator: c});
                return null;
                break
              }
            }
          } else return null;
        }
      }
      if (this.props.value2Filter && this.state.value1 !== "") {
        const re = RegExp(this.props.value2Filter);
        if (!re.test(this.state.value2) && !re.test(this.state.value1) && !re.test(this.state.comparator)) return null;
      }
      let value1_outputs_strs = [];
      let found_values = false;
      if (this.state.value1_outputs) {
        for (const k in this.state.value1_outputs) {
          // console.log(k, this.state.value1_outputs)
          if (this.state.value1_outputs[k] && this.state.value1_outputs[k].length == 1) {
            // console.log(this.state.value1_outputs[k])
            const d = this.state.value1_outputs[k][0][1];
            let str = k + " model outputs: \n";
            for (const name in d) {
              // console.log("d[name]", d[name])
              if (typeof d[name] === 'string') {
                str += name + ": " + "|".join(d[name].split("|").map(x => "" + parseFloat(x).toFixed(3)));
              } else {
                str += name + ": " + d[name].toFixed(3) + "\n";
              }
              
              found_values = true;
            }
            value1_outputs_strs.push(str);
          }
        }
      }
      if (found_values) {
        value1_outputs_str = value1_outputs_strs.join("\n");
      }

      let value2_outputs_strs = [];
      found_values = false;
      if (this.state.value2_outputs) {
        for (const k in this.state.value2_outputs) {
          // console.log(k, this.state.value2_outputs)
          if (this.state.value2_outputs[k] && this.state.value2_outputs[k].length == 1) {
            // console.log(this.state.value2_outputs[k])
            const d = this.state.value2_outputs[k][0][1];
            let str = k + " model outputs: \n";
            for (const name in d) {
              // console.log("d[name]", d[name])
              str += name + ": " + d[name].toFixed(3) + "\n";
              found_values = true;
            }
            value2_outputs_strs.push(str);
          }
        }
      }
      if (found_values) {
        value2_outputs_str = value2_outputs_strs.join("\n");
      }


    } else if (this.props.value2Filter) {
      const re = RegExp(this.props.value2Filter); // TODO: rename value2Filter to reflect it's global nature
      if (!re.test(this.state.topic_name)) return null;
    }

    // let score = {};
    // if (this.state.scores) {
    //   for (let k in this.state.scores) {
    //     score[k] = Math.max(...this.state.scores[k].map(x => x[1]));
    //   }
    // } else {
    //   for (let k in this.state.scores) {
    //     score[k] = NaN;
    //   }
    // }

    

    let outerClasses = "gadfly-row-child";
    if (this.props.selected) outerClasses += " gadfly-row-selected";
    if (this.state.dropHighlighted) outerClasses += " gadfly-row-drop-highlighted";
    if (this.state.dragging) outerClasses += " gadfly-row-dragging";
    if (this.props.isSuggestion && this.state.plusHovering) outerClasses += " gadfly-row-hover-highlighted";
    //if (this.state.hidden) outerClasses += " gadfly-row-hidden";

    let hideClasses = "gadfly-row-hide-button";
    if (this.state.hovering) hideClasses += " gadfly-row-hide-hovering";
    if (this.state.hidden) hideClasses += " gadfly-row-hide-hidden";

    let addTopicClasses = "gadfly-row-hide-button";
    if (this.state.hovering) addTopicClasses += " gadfly-row-hide-hovering";

    let editRowClasses = "gadfly-row-hide-button";
    if (this.state.hovering) editRowClasses += " gadfly-row-hide-hovering";
    if (this.state.editing) editRowClasses += " gadfly-row-hide-hidden";
    
    let overall_score = {};
    if (this.state.scores) {
      for (let k in this.state.scores) {
        overall_score[k] = Math.max(...this.state.scores[k].filter(x => Number.isFinite(x[1])).map(x => x[1]));
      }
    } else {
      for (let k in this.state.scores) {
        overall_score[k] = NaN;
      }
    }

    if (this.props.scoreFilter && overall_score[main_score] < this.props.scoreFilter) {
      //console.log("score filter ", this.state.value1, score, this.props.scoreFilter)
      return null;
    }

    // console.log("about to return code from Row render...", this.state, this.state.scores, overall_score, this.props.scoreColumns)

    return <div className={outerClasses} draggable onMouseOver={this.onMouseOver} onMouseOut={this.onMouseOut} onMouseDown={this.onMouseDown}
                onDragStart={this.onDragStart} onDragEnd={this.onDragEnd} onDragOver={this.onDragOver}
                onDragEnter={this.onDragEnter} onDragLeave={this.onDragLeave} onDrop={this.onDrop} ref={(el) => this.divRef = el}
                style={this.props.hideBorder ? {} : {borderTop: "1px solid rgb(216, 222, 228)"}} tabindex="0" onKeyDown={this.keyDownHandler}>
      <ContextMenu top={this.state.contextTop} left={this.state.contextLeft} open={this.state.contextOpen}
                   onClose={this.closeContextMenu} rows={this.state.contextRows} onClick={this.handleContextMenuClick} />
      {/* {!this.props.isSuggestion && !this.props.hideButtons &&
        <div onClick={this.toggleEditRow} className={editRowClasses} style={{marginLeft: "-60px", marginRight: "5px", cursor: "pointer"}}>
          {(this.state.topic_name !== null || true) &&
            <FontAwesomeIcon icon={faEdit} style={{fontSize: "14px", color: "#000000", display: "inline-block"}} />
          }
        </div>
      } */}
      {/* {!this.props.isSuggestion && !this.props.hideButtons &&
        <div onClick={this.addNewTopic} className={addTopicClasses} style={{marginLeft: "-25px", marginRight: "5px", cursor: "pointer"}}>
          <FontAwesomeIcon icon={faFolderPlus} style={{fontSize: "14px", color: "#000000", display: "inline-block"}} title="Create new sub-topic" />
        </div>
      } */}
      {/* {!this.props.isSuggestion &&
        <div onClick={this.toggleHideTopic} className={hideClasses} style={{marginLeft: "0px", marginRight: "0px", cursor: "pointer"}}>
          <FontAwesomeIcon icon={faEyeSlash} style={{fontSize: "14px", color: "#000000", display: "inline-block"}} />
        </div>
      } */}
      {this.state.topic_name !== null &&
        <div onClick={this.onOpen} class="gadfly-row-add-button" style={{marginLeft: "6px", lineHeight: "14px", opacity: "1", cursor: "pointer", paddingLeft: "4px", marginRight: "3px", paddingRight: "0px", display: "inline-block"}}>
          <FontAwesomeIcon icon={faFolder} style={{fontSize: "14px", color: "rgb(84, 174, 255)", display: "inline-block"}} />
        </div>
      }
      {this.props.isSuggestion &&
        <div onClick={this.addToCurrentTopic} className="gadfly-row-add-button gadfly-hover-opacity" style={{cursor: "pointer"}} onMouseOver={this.onPlusMouseOver} onMouseOut={this.onPlusMouseOut}>
          <FontAwesomeIcon icon={faPlus} style={{fontSize: "14px", color: "#000000", display: "inline-block"}} title="Add to current topic" />
        </div>
      }
      <div style={{padding: "5px", flex: 1}} onClick={this.clickRow} onDoubleClick={this.onOpen}>  
        {this.state.topic_name !== null ? [
          <div style={{display: "flex", marginTop: "3px", fontSize: "14px"}}> 
            <div className={this.state.hidden ? "gadfly-row-hidden": ""} style={{flex: "1", textAlign: "left"}}>
              <ContentEditable onClick={this.clickTopicName} ref={el => this.topicNameEditable = el} text={this.state.topic_name} onInput={this.inputTopicName} onFinish={this.finishTopicName} editable={this.state.editing} />
            </div>
          </div>,
          <div className="gadfly-row" style={{opacity: 0.6, marginTop: "-16px", display: this.state.previewValue1 ? 'flex' : 'none'}}>
            {/* <div style={{flex: "0 0 140px", textAlign: "left"}}>
              <span style={{color: "#aaa"}}>{this.state.prefix}</span>
            </div> */}
            <div className="gadfly-row-input">
              <span style={{color: "#aaa", opacity: this.state.hovering ? 1 : 0, transition: "opacity 1s"}}>{this.state.prefix}</span><span style={{color: "#aaa"}}>"</span>{this.state.previewValue1}<span style={{color: "#aaa"}}>"</span>
            </div>
            <div style={{flex: "0 0 "+this.props.selectWidth+"px", color: "#999999", textAlign: "center", overflow: "hidden", opacity: (this.state.previewValue1 ? 1 : 0)}}>
              <div style={{lineHeight: "13px", height: "16px", opacity: "1.0", verticalAlign: "middle", display: "inline-block"}}>
                {/* <FontAwesomeIcon icon={faArrowRight} style={{fontSize: "14px", color: "#000000", display: "inline-block"}} /> */}
                {/* <svg version="1.1" x="0px" y="0px" viewBox="0 0 523.06 513.85">
                  <path style={{fill: "#000000"}} d="M227.71,67.82l22.2-22.2c9.4-9.4,24.6-9.4,33.9,0l194.4,194.3c9.4,9.4,9.4,24.6,0,33.9l-194.4,194.4
                    c-9.4,9.4-24.6,9.4-33.9,0l-22.2-22.2c-9.5-9.5-9.3-25,0.4-34.3l120.5-114.8H61.21c-13.3,0-24-10.7-24-24v-32c0-13.3,10.7-24,24-24
                    h287.4l-120.5-114.8C218.31,92.82,218.11,77.32,227.71,67.82z"/>
                  <g>
                    <path style={{fill: "#000000"}} d="M481,502.85c-5.58,0-10.94-2.24-15.11-6.31L18.8,67.45c-8.85-8.69-10.32-23.84-3.36-34.5l7.47-11.46
                      C27.19,14.93,34.12,11,41.46,11c5.58,0,10.94,2.24,15.12,6.32l447.08,429.08c8.85,8.68,10.33,23.83,3.38,34.5l-7.48,11.46
                      C495.27,498.93,488.33,502.85,481,502.85C481,502.85,481,502.85,481,502.85z"/>
                    <path style={{fill: "#FFFFFF"}} d="M41.46,22c2.62,0,5.26,1.05,7.47,3.22l447.03,429.02c5.17,5.07,6,14.31,1.87,20.64l-7.48,11.46
                      c-2.37,3.62-5.84,5.5-9.35,5.5c-2.62,0-5.26-1.05-7.46-3.22L26.5,59.61c-5.16-5.07-6-14.31-1.86-20.64l7.48-11.46
                      C34.48,23.88,37.95,22,41.46,22 M41.46,0C30.39,0,20.01,5.79,13.69,15.49L6.22,26.95c-9.88,15.13-7.79,35.92,4.87,48.35l0.09,0.09
                      l0.09,0.09l446.96,428.96c6.24,6.07,14.32,9.41,22.77,9.41c11.05,0,21.43-5.78,27.76-15.46l7.49-11.48
                      c9.88-15.16,7.77-35.95-4.89-48.36l-0.08-0.08l-0.09-0.08L64.23,9.41C57.99,3.34,49.91,0,41.46,0L41.46,0z"/>
                  </g>
                </svg> */}
                {/* <svg version="1.1" x="0px" y="0px" viewBox="0 0 448 512" style={{height: "100%"}}>
                  <path style={{fill: "#000000"}} d="M416,208c17.67,0,32-14.33,32-32v-32c0-17.67-14.33-32-32-32h-23.88l51.87-66.81
                    c5.37-7.02,4.04-17.06-2.97-22.43L415.61,3.3c-7.02-5.38-17.06-4.04-22.44,2.97L311.09,112H32c-17.67,0-32,14.33-32,32v32
                    c0,17.67,14.33,32,32,32h204.56l-74.53,96H32c-17.67,0-32,14.33-32,32v32c0,17.67,14.33,32,32,32h55.49l-51.87,66.81
                    c-5.37,7.01-4.04,17.05,2.97,22.43L64,508.7c7.02,5.38,17.06,4.04,22.43-2.97L168.52,400H416c17.67,0,32-14.33,32-32v-32
                    c0-17.67-14.33-32-32-32H243.05l74.53-96H416z"/>
                </svg> */}
                <span style={{color: "#aaa"}}>should not be</span> {/* TODO: fix this for varying comparators */}
              </div>
            </div>
            <div style={{flex: "0 0 200px", textAlign: "left"}}>
            <span style={{color: "#aaa"}}>"</span>{this.state.previewValue2}<span style={{color: "#aaa"}}>"</span>
            </div>
          </div>
          //onDoubleClick={this.onOpen} 
          
         ] : (
          <div className="gadfly-row">
            {/* <div style={{flex: "0 0 160px", textAlign: "right"}}>
              <span style={{color: "#aaa"}}>The model output for&nbsp;</span>
            </div> */}
            <div className={"gadfly-row-input" + (this.state.hidden ? " gadfly-row-hidden": "")} onClick={this.clickRow}>
              <span style={{color: "#aaa", opacity: this.state.hovering ? 1 : 0, transition: "opacity 1s"}}>{!this.props.inFillin && this.state.prefix}</span
              >&nbsp;<div onClick={this.clickValue1} style={{display: "inline-block"}}><span style={{color: "#aaa"}}>"</span><span title={value1_outputs_str} onContextMenu={this.handleValue1ContextMenu}><ContentEditable onClick={this.clickValue1} onTemplateExpand={this.templateExpandValue1} ref={el => this.value1Editable = el} text={this.state.value1} onInput={this.inputValue1} onFinish={this.finishValue1} editable={this.state.editing} defaultText={this.props.value1Default} /></span><span style={{color: "#aaa"}}>"</span></div>
            </div>
            {!this.props.inFillin && <div style={{flex: "0 0 "+this.props.selectWidth+"px", color: "#999999", textAlign: "center", overflow: "hidden", display: "flex"}}>
              <div style={{alignSelf: "flex-end", display: "inline-block"}}>
                <span style={{color: "#aaa"}}><select class="gadfly-plain-select" style={{marginLeft: "4px", color: "#aaa"}} value={this.state.comparator} onChange={this.changeComparator}>
                    <option>should not be</option>
                    <option>should be</option>
                    <option>should be the same as for</option>
                    <option>should be invertable.</option>
                    {/* <option>should not be less than for</option> */}
                  </select></span>
              </div>
            </div>}
            {!this.props.inFillin && <div className={this.state.hidden ? "gadfly-row-hidden": ""} onClick={this.clickValue2} style={{maxWidth: "400px", overflowWrap: "anywhere", flex: "0 0 200px", textAlign: "left", display: "flex"}}>
              <span style={{alignSelf: "flex-end"}}>
                {this.state.comparator === "should be invertable." ?
                  <span><span style={{color: "#aaa"}}>"</span><span style={{color: "#666666"}}>{this.state.value2}</span><span style={{color: "#aaa"}}>"</span><span style={{color: "#aaa", opacity: this.state.hovering ? 1 : 0, transition: "opacity 1s"}}>&nbsp;{!this.props.inFillin && "is the inversion."}</span></span>
                :
                  <span><span style={{color: "#aaa"}}>"</span><span title={value2_outputs_str}><ContentEditable ref={el => this.value2Editable = el} onClick={this.clickValue2} text={this.state.value2} onInput={this.inputValue2} onFinish={_ => this.setState({editing: false})} editable={this.state.editing} defaultText={this.props.value2Default} /></span><span style={{color: "#aaa"}}>"</span></span>
                }
              </span>
            </div>}
          </div>
        )}
      </div>
      {/* <div className="gadfly-row-score-text-box">
        {this.state.topic_name === null && !isNaN(score) && score.toFixed(3).replace(/\.?0*$/, '')}
      </div> */}
      {this.props.scoreColumns && this.props.scoreColumns.map(k => {
        if (Number.isFinite(overall_score[k]) && this.props.updateTotals) {
          this.props.updateTotals(
            this.state.scores[k].reduce((total, value) => total + (value[1] <= 0), 0),
            this.state.scores[k].reduce((total, value) => total + (value[1] > 0), 0)
          );
        }
        
        // this.totalPasses[k] = Number.isFinite(overall_score[k]) ? this.state.scores[k].reduce((total, value) => total + (value[1] <= 0), 0) : NaN;
        // this.totalFailures[k] = this.state.scores[k].reduce((total, value) => total + (value[1] > 0), 0);
        return <div className="gadfly-row-score-plot-box">
          {overall_score[k] > 0 ?
            <svg height="20" width="100">
              {Number.isFinite(overall_score[k]) && [
                <line x1="50" y1="10" x2={50 + 48*scale_score(overall_score[k])} y2="10" style={{stroke: "rgba(0, 0, 0, 0.1)", strokeWidth: "20"}}></line>,
                this.state.scores[k].filter(x => Number.isFinite(x[1])).map((score, index) => {
                  //console.log("scale_score(score[1])", scale_score(score[1]))
                  return <line onMouseOver={e => this.onScoreOver(e, score[0])}
                              onMouseOut={e => this.onScoreOut(e, score[0])}
                              x1={50 + 48*scale_score(score[1])} y1="0"
                              x2={50 + 48*scale_score(score[1])} y2="20"
                              style={{stroke: score[1] <= 0 ? "rgb(26, 127, 55)" : "rgb(207, 34, 46)", strokeWidth: "2"}}
                        ></line>
                }),
                <line x1={50} y1="0"
                      x2={50} y2="20" stroke-dasharray="2"
                      style={{stroke: "#bbbbbb", strokeWidth: "1"}}
                ></line>,
                this.state.topic_name !== null && <text x="25" y="11" dominant-baseline="middle" text-anchor="middle" style={{transition: "fill-opacity 1s, stroke-opacity 1s", strokeOpacity: this.state.hovering*1, fillOpacity: this.state.hovering*1, pointerEvents: "none", fill: "#ffffff", fontSize: "11px", strokeWidth: "3px", stroke: "rgb(26, 127, 55)", opacity: 1, strokeLinecap: "butt", strokeLinejoin: "miter", paintOrder: "stroke fill"}}>{this.state.scores[k].reduce((total, value) => total + (value[1] <= 0), 0)}</text>,
                this.state.topic_name !== null && <text x="75" y="11" dominant-baseline="middle" text-anchor="middle" style={{transition: "fill-opacity 1s, stroke-opacity 1s", strokeOpacity: this.state.hovering*1, fillOpacity: this.state.hovering*1, pointerEvents: "none", fill: "#ffffff", fontSize: "11px", strokeWidth: "3px", stroke: "rgb(207, 34, 46)", opacity: 1, strokeLinecap: "butt", strokeLinejoin: "miter", paintOrder: "stroke fill"}}>{this.state.scores[k].reduce((total, value) => total + (value[1] > 0), 0)}</text>,
                this.state.topic_name === null && <text x="75" y="11" dominant-baseline="middle" text-anchor="middle" style={{transition: "fill-opacity 1s, stroke-opacity 1s", strokeOpacity: this.state.hovering*1, fillOpacity: this.state.hovering*1, pointerEvents: "none", fill: "#ffffff", fontSize: "11px", strokeWidth: "3px", stroke: "rgb(207, 34, 46)", opacity: 1, strokeLinecap: "butt", strokeLinejoin: "miter", paintOrder: "stroke fill"}}>{overall_score[k].toFixed(3).replace(/\.?0*$/, '')}</text>,
                this.state.topic_name === 3324 && !isNaN(overall_score[k]) && <text x={(48*scale_score(overall_score[k]) > 3000 ? 50 + 5 : 50 + 48*scale_score(overall_score[k]) + 5)} y="11" dominant-baseline="middle" text-anchor="start" style={{pointerEvents: "none", fontSize: "11px", opacity: 0.7, fill: "rgb(207, 34, 46)"}}>{overall_score[k].toFixed(3).replace(/\.?0*$/, '')}</text>
              ]}
            </svg>
          :
            <svg height="20" width="100">
              {Number.isFinite(overall_score[k]) && [
                <line x2="50" y1="10" x1={50 + 48*scale_score(overall_score[k])} y2="10" style={{stroke: "rgba(0, 0, 0, 0.1)", strokeWidth: "20"}}></line>,
                this.state.scores[k].filter(x => Number.isFinite(x[1])).map((score, index) => {
                  return <line onMouseOver={e => this.onScoreOver(e, score[0])}
                              onMouseOut={e => this.onScoreOut(e, score[0])}
                              x1={50 + 48*scale_score(score[1])} y1="0"
                              x2={50 + 48*scale_score(score[1])} y2="20"
                              style={{stroke: score[1] <= 0 ? "rgb(26, 127, 55)" : "rgb(207, 34, 46)", strokeWidth: "2"}}
                        ></line>
                }),
                <line x1={50} y1="0"
                      x2={50} y2="20" stroke-dasharray="2"
                      style={{stroke: "#bbbbbb", strokeWidth: "1"}}
                ></line>,
                this.state.topic_name !== null && <text x="25" y="11" dominant-baseline="middle" text-anchor="middle" style={{transition: "fill-opacity 1s, stroke-opacity 1s", strokeOpacity: this.state.hovering*1, fillOpacity: this.state.hovering*1, pointerEvents: "none", fill: "#ffffff", fontSize: "11px", strokeWidth: "3px", stroke: "rgb(26, 127, 55)", opacity: 1, strokeLinecap: "butt", strokeLinejoin: "miter", paintOrder: "stroke fill"}}>100%</text>,
                this.state.topic_name !== null && <text x="75" y="11" dominant-baseline="middle" text-anchor="middle" style={{transition: "fill-opacity 1s, stroke-opacity 1s", strokeOpacity: this.state.hovering*1, fillOpacity: this.state.hovering*1, pointerEvents: "none", fill: "#ffffff", fontSize: "11px", strokeWidth: "3px", stroke: "rgb(207, 34, 46)", opacity: 1, strokeLinecap: "butt", strokeLinejoin: "miter", paintOrder: "stroke fill"}}>0%</text>,
                this.state.topic_name === null && <text x="25" y="11" dominant-baseline="middle" text-anchor="middle" style={{transition: "fill-opacity 1s, stroke-opacity 1s", strokeOpacity: this.state.hovering*1, fillOpacity: this.state.hovering*1, pointerEvents: "none", fill: "#ffffff", fontSize: "11px", strokeWidth: "3px", stroke: "rgb(26, 127, 55)", opacity: 1, strokeLinecap: "butt", strokeLinejoin: "miter", paintOrder: "stroke fill"}}>{overall_score[k].toFixed(3).replace(/\.?0*$/, '')}</text>,
                this.state.topic_name === 2342 && !isNaN(overall_score[k]) && <text x={(48*scale_score(overall_score[k]) < -3000 ? 50 - 5 : 50 + 48*scale_score(overall_score[k]) - 5)} y="11" dominant-baseline="middle" text-anchor="end" style={{pointerEvents: "none", fontSize: "11px", opacity: 0.7, fill: "rgb(26, 127, 55)"}}>{overall_score[k].toFixed(3).replace(/\.?0*$/, '')}</text>
              ]}
            </svg>
          }
        </div>
      })}
    </div>
  }

  onMouseDown(e) {
    this.mouseDownTarget = e.target;
  }

  handleContextMenuClick(row) {
    console.log("handleContextMenuClick", row)
    if (row === "Expand into a template") {
      console.log("EXPAND!!", this.props.id, this.state.contextFocus);
      if (this.state.contextFocus === "value1") {
        this.props.comm.send(this.props.id, {"action": "template_expand_value1"});
      }
    }
    this.setState({contextOpen: false});
  }

  closeContextMenu() {
    this.setState({contextOpen: false});
  }

  handleValue1ContextMenu(e) {
    e.preventDefault();
    console.log("handleValue1ContextMenu open", e, e.pageY, e.pageX)
    this.setState({contextTop: e.pageY, contextLeft: e.pageX, contextOpen: true, contextFocus: "value1", contextRows: ["Expand into a template"]});
  }

  templateExpandValue1() {
    console.log("templateExpandValue1")
    this.props.comm.send(this.props.id, {"action": "template_expand_value1"});
  }

  keyDownHandler(e) {
    if (e.keyCode == 13) {
      console.log("return!", this.props.soleSelected, this.props.selected)
      if (this.props.soleSelected) {
        if (this.state.topic_name !== null) {
          this.onOpen(e);
        } else if (this.props.isSuggestion) {
          this.addToCurrentTopic(e);
          this.doOnNextDataLoad(() => this.props.giveUpSelection(this.props.id));
        }
      }
    }
  }

  doOnNextDataLoad(f) {
    this.dataLoadActions.push(f);
  }

  changeComparator(e) {
    this.props.comm.send(this.props.id, {"comparator": e.target.value});
    this.setState({comparator: e.target.value});
  }

  onScoreOver(e, key) {
    this.setState({
      previewValue1: this.props.comm.data[key].value1,
      previewValue2: this.props.comm.data[key].value2
    })
  }
  onScoreOut(e, key) {
    this.setState({
      previewValue1: null,
      previewValue2: null
    })
  }

  toggleEditRow(e) {
    e.preventDefault();
    e.stopPropagation();

    if (!this.state.editing) {
      this.setState({editing: true});
      console.log("about to edit focus")
      if (this.state.topic_name === null) {
        defer(() => this.value1Editable.focus());
      } else {
        defer(() => this.topicNameEditable.focus());
      }
    } else {
      this.setState({editing: false});
    }
  }

  toggleHideTopic(e) {
    e.preventDefault();
    e.stopPropagation();

    this.props.comm.send(this.props.id, {"hidden": !this.state.hidden});
  }

  /* addNewTopic(e) {
    e.preventDefault();
    e.stopPropagation();
    const newName = this.props.generateTopicName();
    const newTopic = this.props.topic + "/" + newName;
    if (this.state.topic_name === null) {
      this.props.comm.send(this.props.id, {"topic": newTopic });
    } else {
      this.props.comm.send(this.props.id, {"topic": newTopic + "/" + this.state.topic_name});
    }
    this.props.setSelected(newTopic);
  } */

  onMouseOver(e) {
    //console.log("onMouseOver")
    //e.preventDefault();
    //e.stopPropagation();
    this.setState({hovering: true});
  }
  onMouseOut(e) {
    //e.preventDefault();
    //e.stopPropagation();
    this.setState({hovering: false});
  }

  onPlusMouseOver(e) {
    //console.log("onPlusMouseOver")
    //e.preventDefault();
    //e.stopPropagation();
    this.setState({plusHovering: true});
  }
  onPlusMouseOut(e) {
    //e.preventDefault();
    //e.stopPropagation();
    this.setState({plusHovering: false});
  }

  onOpen(e) {
    e.preventDefault();
    e.stopPropagation();
    console.log("Row.onOpen(", e, ")");
    if (this.state.topic_name !== null && this.props.onOpen) {
      this.props.onOpen(this.props.topic + "/" + this.state.topic_name);
    }
  }

  inputValue1(text) {
    console.log("inputValue1", text)
    this.setState({value1: text, scores: null});
    this.props.comm.debouncedSend500(this.props.id, {value1: text});
  }

  finishValue1(text) {
    console.log("finishValue1", text)
    this.setState({editing: false});
    if (text.includes("/")) {
      this.setState({value1: text, scores: null});
      this.props.comm.send(this.props.id, {value1: text});
    }
  }

  inputValue2(text) {
    console.log("inputValue2", text)
    if (this.props.value2Edited) {
      this.props.value2Edited(this.props.id, this.state.value2, text);
    }
    this.setValue2(text);
  }

  setValue2(text) {
    this.setState({value2: text, scores: null});
    this.props.comm.debouncedSend500(this.props.id, {value2: text});
  }

  inputTopicName(text) {
    this.setState({topic_name: text.replace("\\", "").replace("\n", "")});
  }

  finishTopicName(text) {
    console.log("finishTopicName", text)
    
    this.setState({topic_name: text.replace("\\", "").replace("\n", ""), editing: false});
    this.props.comm.send(this.props.id, {topic: this.props.topic + "/" + text});
  }
  
  clickRow(e) {
    const modKey = e.metaKey || e.shiftKey;
    if (this.props.onSelectToggle) {
      e.preventDefault();
      e.stopPropagation();
      this.props.onSelectToggle(this.props.id, e.shiftKey, e.metaKey);
    }
  }

  clickTopicName(e) {
    console.log("clickTopicName");
    const modKey = e.metaKey || e.shiftKey;
    if (this.props.onSelectToggle) {
      e.preventDefault();
      e.stopPropagation();
      this.props.onSelectToggle(this.props.id, e.shiftKey, e.metaKey);
    }
    if (!modKey && !this.state.editing) {
      this.setState({editing: true});
      console.log("topic editing", this.state.editing)
      e.preventDefault();
      e.stopPropagation();
      defer(() => this.topicNameEditable.focus());
    }
  }

  clickValue1(e) {
    console.log("clickValue1", e);
    const modKey = e.metaKey || e.shiftKey;
    if (this.props.onSelectToggle) {
      e.preventDefault();
      e.stopPropagation();
      this.props.onSelectToggle(this.props.id, e.shiftKey, e.metaKey);
    }
    if (!modKey && !this.state.editing) {
      this.setState({editing: true});
      console.log("value1 editing", this.state.editing)
      e.preventDefault();
      e.stopPropagation();
      defer(() => this.value1Editable.focus());
    }
  }

  

  clickValue2(e) {
    console.log("clickValue2");
    const modKey = e.metaKey || e.shiftKey;
    if (this.props.onSelectToggle) {
      e.preventDefault();
      e.stopPropagation();
      this.props.onSelectToggle(this.props.id, e.shiftKey, e.metaKey);
    }
    if (!modKey && !this.state.editing) {
      this.setState({editing: true});
      e.preventDefault();
      e.stopPropagation();
      defer(() => this.value2Editable.focus());
    }
  }

  onDragStart(e) {

    // don't initiate a drag from inside an editiable object
    if (this.mouseDownTarget.getAttribute("contenteditable") === "true") {
      e.preventDefault();
      return false;
    }
    //console.log("drag start", e, this.mouseDownTarget.getAttribute("contenteditable") === "true")
    this.setState({dragging: true});
    e.dataTransfer.setData("id", this.props.id);
    e.dataTransfer.setData("topic_name", this.state.topic_name);
    if (this.props.onDragStart) {
      this.props.onDragStart(e, this);
    }
  }

  onDragEnd(e) {
    this.setState({dragging: false});
    if (this.props.onDragEnd) {
      this.props.onDragEnd(e, this);
    }
  }

  onDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  onDragEnter(e) {
    console.log("enter", e.target)
    e.preventDefault();
    e.stopPropagation();
    if (this.state.topic_name !== null) {
      this.setState({dropHighlighted: this.state.dropHighlighted + 1});
    }
  }

  onDragLeave(e) {
    console.log("leave", e.target)
    e.preventDefault();
    e.stopPropagation();
    if (this.state.topic_name !== null) {
      this.setState({dropHighlighted: this.state.dropHighlighted - 1});
    }
  }

  onDrop(e) {
    
    const id = e.dataTransfer.getData("id");
    const topic_name = e.dataTransfer.getData("topic_name");
    if (this.state.topic_name !== null) {
      this.setState({dropHighlighted: 0});
      if (this.props.onDrop && id !== this.props.id) {
        if (topic_name !== null && topic_name !== "null") {
          this.props.onDrop(id, {topic: this.props.topic + "/" + this.state.topic_name + "/" + topic_name});
        } else {
          this.props.onDrop(id, {topic: this.props.topic + "/" + this.state.topic_name});
        }
      }
    }
  }

  addToCurrentTopic(e) {
    e.preventDefault();
    e.stopPropagation();
    console.log("addToCurrentTopic");
    this.props.comm.send(this.props.id, {topic: this.props.topic});
  }
}

class BreadCrum extends React.Component {
  constructor(props) {
    super(props);
    autoBind(this);
    
    this.state = {
      dropHighlighted: 0
    };
  }

  render() {
    // console.log("br", this.props.name, this.props.name === "")
    return <div className={this.state.dropHighlighted ? "gadfly-crum-selected" : ""} style={{borderRadius: "10px 10px 10px 10px", display: "inline-block", cursor: "pointer"}} 
         onClick={this.onClick} onDragOver={this.onDragOver} onDragEnter={this.onDragEnter}
         onDragLeave={this.onDragLeave} onDrop={this.onDrop}>
      {this.props.name === "" ? "Tests" : this.props.name}
    </div>
  }

  onClick(e) {
    e.preventDefault();
    e.stopPropagation();
    if (this.props.onClick) {
      if (this.props.name === "") {
        this.props.onClick(this.props.topic);
      } else {
        this.props.onClick(this.props.topic + "/" + this.props.name);
      }
    }
  }

  onDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  onDragEnter(e) {
    e.preventDefault();
    e.stopPropagation();
    this.setState({dropHighlighted: this.state.dropHighlighted + 1});
  }

  onDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    this.setState({dropHighlighted: this.state.dropHighlighted - 1});
  }

  onDrop(e) {
    const id = e.dataTransfer.getData("id");
    this.setState({dropHighlighted: 0});
    if (this.props.onDrop) {
      let suffix = "";
      if (id.includes("/")) {
        suffix = "/" + id.split("/").pop();
      }
      this.props.onDrop(id, {topic: this.props.topic + (this.props.name === "" ? "" : "/" + this.props.name) + suffix});
    }
  }
}

class TotalValue extends React.Component {
  constructor(props) {
    super(props);
    autoBind(this);

    // our starting state 
    this.state = {
      // note that all the ids will also be properties of the state
    };
  }

  setSubtotal(id, subtotal) {
    let update = {};
    update[id] = subtotal;
    this.setState(update);
  }

  render() {
    // we just sum up the current active subtotals
    let total = 0;
    for (let i in this.props.activeIds) {
      total += get(this.state, this.props.activeIds[i], 0);
    }
    
    return <span>
      {total}
    </span>
  }
}


class IOPairChart extends React.Component {
  constructor(props) {
    super(props);
    autoBind(this);

    // our starting state 
    this.state = {
      topic: "/",
      suggestions: [],
      tests: [],
      selections: {},
      loading_suggestions: false,
      max_suggestions: 10,
      suggestions_pos: 0,
      suggestionsDropHighlighted: 0,
      score_filter: 0.3,
      do_score_filter: true,
      experiment_pos: 0,
      timerExpired: false,
      experiment_locations: [],
      experiment: false,
      value2Filter: ""
    };

    console.log("this.props.location", this.props.location)

    this.id = 'test_chart';

    this.rows = {};

    // connect to the jupyter backend
    console.log("pairs this.props.interfaceId", this.props.interfaceId)
    if (this.props.environment === "jupyter") {
      this.comm = new GadflyJupyterComm(this.props.interfaceId, this.connectionOpen);
    } else if (this.props.environment === "web") {
      this.comm = new GadflyWebSocketComm(this.props.interfaceId, this.props.websocket_server, this.connectionOpen);
    } else {
      console.error("Unknown environment:", this.props.environment);
    }
    this.comm.subscribe(this.id, this.newData);

    this.debouncedForceUpdate = debounce(this.debouncedForceUpdate, 100);

    window.pair_chart = this;
  }

  debouncedForceUpdate() {
    this.forceUpdate();
  }

  // gets called once we have a working websocket connection ready to go
  connectionOpen() {
    // if we are in Jupyter we need to sync the URL in the MemoryRouter
    if (this.props.environment == "jupyter") {
      this.props.history.push(this.props.prefix + this.props.startingTopic);

    // if we don't have a starting topic then we are stand-alone and need to sync our state with the address bar
    } else if (this.props.location.pathname !== (this.props.prefix == "" ? "/" : this.props.prefix)) {
      defer(() => this.goToTopic(this.stripPrefix(this.props.location.pathname)));
    }
    this.props.history.listen(this.locationChanged);

    defer(() => this.comm.send(this.id, {action: "redraw"}));
  }

  stripPrefix(path) {
    if (path.startsWith(this.props.prefix)) {
      return path.slice(this.props.prefix.length);
    } else {
      return path;
    }
  }

  componentDidUpdate() {
    this.componentDidUpdateOrMount(false);
  }

  componentDidMount() {
    this.componentDidUpdateOrMount(true);
  }
  
  componentDidUpdateOrMount(mount) {
    // console.log("Row componentDidUpdateOrMount", mount, this.state.selections)
    if (Object.keys(this.state.selections).length === 0) {
      // this.divRef.focus();
    }
  }

  render() {

    const selectWidths = {
      "should not be": 88,
      "should be": 65,
      "should be the same as for": 161,
      "should be invertable.": 141
    }
    let maxSelectWidth = Math.max(
      ...this.state.suggestions.map(id => this.comm.data[id] ? get(selectWidths, this.comm.data[id].comparator, 40) : 40),
      ...this.state.tests.map(id => this.comm.data[id] ? get(selectWidths, this.comm.data[id].comparator, 40) : 40)
    );
    if ("suggestions_template_row" in this.comm.data) {
      maxSelectWidth = Math.max(get(selectWidths, this.comm.data["suggestions_template_row"].comparator, 40), maxSelectWidth);
    }
    // const location = useLocation();

    const inFillin = this.state.topic.startsWith("/Fill-ins");

    // console.log("location.pathname", location.pathname);

    let topicPath = "";
    // console.log("tests.render4", this.state.tests, stripSlash(this.stripPrefix(this.props.location.pathname)), this.state.topic);

    let breadCrumbParts = stripSlash(this.stripPrefix(this.state.topic)).split("/");

    let totalPasses = <TotalValue activeIds={this.state.tests} ref={(el) => this.totalPassesObj = el} />;
    let totalFailures = <TotalValue activeIds={this.state.tests} ref={(el) => this.totalFailuresObj = el} />;

    return (<div onKeyDown={this.keyDownHandler} tabindex="0" style={{outline: "none"}} ref={(el) => this.divRef = el}>
      {this.state.timerExpired && <div style={{fontSize: "20px", paddingTop: "100px", color: "#ffffff", background: "#880000", position: "absolute", top: "0px", left: "0px", width: "100%", height: "100%", zIndex: "1000", opacity: 0.9, textAlign: "center", verticalAlign: "middle"}}>
        This task's time period is done. Now transitioning to the next tesk...
      </div>}
      {this.state.experiment && <div>
        <Clock duration={this.state.experiment_locations[this.state.experiment_pos].duration} onFinish={this.clockFinished} />
      </div>}
      {this.props.checklistMode && <div style={{textAlign: "right"}}>
        <div onClick={this.openTests} style={{
             cursor: "pointer", boxSizing: "border-box", display: "inline-block",
             color: !inFillin ? "#ffffff" : "#999999",
             borderRadius: "7px 0px 0px 7px", marginRight: "auto", marginLeft: "auto",
             marginBottom: "0px", background: !inFillin ? "#666666" : "#f5f5f5",
             padding: "8px", paddingTop: "2px", paddingBottom: "2px"}}>
          <span>Tests</span>
        </div>
        <div onClick={this.openFillins} style={{
             cursor: "pointer", boxSizing: "border-box", display: "inline-block",
             color: inFillin ? "#ffffff" : "#999999",
             borderRadius: "0px 7px 7px 0px", marginRight: "auto", marginLeft: "auto", marginBottom: "0px",
             background: inFillin ? "#666666" : "#f5f5f5",
             padding: "8px", paddingTop: "2px", paddingBottom: "2px"}}>
          <span>Fill-ins</span>
        </div>
      </div>}

      <div title="Add a new test" onClick={this.addNewTest} style={{float: "right", padding: "9px 10px 7px 14px", border: "1px solid rgb(208, 215, 222)", cursor: "pointer", display: "inline-block", borderRadius: "7px", marginTop: "16px", background: "rgb(246, 248, 250)"}}>
        <div style={{opacity: "0.6", width: "15px", height: "15px", display: "inline-block"}}><FontAwesomeIcon icon={faPlus} style={{fontSize: "13px", color: "#000000", display: "inline-block"}} /></div>
        {/* <span style={{opacity: "0.6", fontSize: "13px", fontWeight: "bold"}}>&nbsp;New Test</span> */}
      </div>
      <div title="Add a new topic" onClick={this.addNewTopic} style={{float: "right", marginRight: "10px", padding: "9px 10px 7px 14px", cursor: "pointer", border: "1px solid rgb(208, 215, 222)", display: "inline-block", borderRadius: "7px", marginTop: "16px", background: "rgb(246, 248, 250)"}}>
        <div style={{opacity: "0.6", width: "15px", height: "15px", display: "inline-block"}}><FontAwesomeIcon icon={faFolderPlus} style={{fontSize: "13px", color: "#000000", display: "inline-block"}} /></div>
        {/* <span style={{opacity: "0.6", fontSize: "13px", fontWeight: "bold"}}>&nbsp;New Topic</span> */}
      </div>
      <div style={{float: "right", marginRight: "10px", padding: "8px 10px 7px 14px", width: "250px", border: "1px solid rgb(208, 215, 222)", display: "inline-block", borderRadius: "7px", marginTop: "16px", background: "rgb(246, 248, 250)"}}>
        <div style={{opacity: "0.6", width: "15px", height: "15px", display: "inline-block", paddingLeft: "1px", marginRight: "10px"}}><FontAwesomeIcon icon={faFilter} style={{fontSize: "13px", color: "#000000", display: "inline-block"}} /></div>
        <span style={{opacity: "0.6", fontSize: "13px", fontWeight: "normal"}}>
          <ContentEditable defaultText="filter tests" text={this.state.value2Filter} onInput={this.inputValue2Filter} />
        </span>
      </div>
      

      <div style={{paddingTop: '20px', width: '100%', verticalAlign: 'top', textAlign: "center"}}>
        <div style={{textAlign: "left", marginBottom: "0px", paddingLeft: "5px", paddingRight: "5px", marginTop: "5px"}}>
          {this.state.score_columns && this.state.score_columns.slice().reverse().map(k => {
            return <div style={{float: "right", width: "110px", textAlign: "center"}}>
              {k != "score" && <div style={{marginTop: "-20px", marginBottom: "20px", height: "0px", cursor: "pointer"}} onClick={e => this.clickModel(k, e)}>{k.replace(" score", "")}</div>}
              {/* <div style={{float: "right", width: "52px", color: red_blue_100[99], textAlign: "left", boxSizing: "border-box", paddingLeft: "3px"}}>
                Fail
              </div>
              <div style={{float: "right", width: "5px", height: "20px", color: red_blue_100[99], textAlign: "center"}}>
                <svg height="20" width="1">
                  <line x1={0} y1="0"
                        x2={0} y2="20" stroke-dasharray="2"
                        style={{stroke: "#aaaaaa", strokeWidth: "1"}}
                  ></line>
                </svg>
              </div>
              <div style={{float: "right", width: "48px", color: red_blue_100[0], textAlign: "right", paddingRight: "4px"}}>
                Pass
              </div> */}
            </div>
          })}
          {/* <div style={{float: "right", width: "205px", color: "#999999", textAlign: "left"}}>
            Output
          </div>
          <div style={{float: "right", width: "30px"}}>
            &nbsp;
          </div>
          <div style={{float: "right", color: "#999999", textAlign: "right"}}>
            Input
          </div> */}
          <span style={{fontSize: "16px"}}>
          {breadCrumbParts.map((name, index) => {
            //console.log("bread crum", name, index);
            const out = <span style={{color: index === breadCrumbParts.length - 1 ? "black" : "rgb(9, 105, 218)" }}>
              {index > 0 && <span style={{color: "black"}}> / </span>}
              <BreadCrum topic={topicPath} name={name} onDrop={this.onDrop} onClick={this.setLocation} />
            </span>
            if (index !== 0) topicPath += "/";
            topicPath += name;
            return index === 0 && this.props.checklistMode ? undefined : out;
          })}
          </span>
          <div style={{clear: "both"}}></div>
          <div></div>
        </div>
        <div clear="all"></div>

        {/* <div style={{marginTop: "5px", opacity: 0.6}}>
          <div className="gadfly-row-child">
            <div style={{paddingRight: "5px", flex: 1}}>  
              <div className="gadfly-row">
                <div style={{flex: "0 0 150px", textAlign: "left", fontStyle: "italic"}}>
                  <ContentEditable defaultText="filter topics" text={this.state.topicFilter} onInput={this.inputTopicFilter} />
                </div>
                <div className="gadfly-row-input" style={{fontStyle: "italic"}}>
                  <ContentEditable defaultText="left value" text={this.state.value1Filter} onInput={this.inputValue1Filter} />
                </div>
                {!inFillin && <div style={{flex: "0 0 "+maxSelectWidth+"px", textAlign: "center", fontStyle: "italic"}}>
                  <ContentEditable defaultText="comp." ref={el => this.comparatorFilterEditable = el} text={this.state.comparatorFilter} onClick={this.clickComparatorFilter} onInput={this.inputComparatorFilter} />
                </div>}
                {!inFillin && <div style={{flex: "0 0 150px", textAlign: "left", fontStyle: "italic"}}>
                  <ContentEditable defaultText="right value" text={this.state.value2Filter} onInput={this.inputValue2Filter} />
                </div>}
              </div>

              

            </div>
            <div className="gadfly-row-score-text-box">
              
            </div>
            <div className="gadfly-row-score-plot-box">
              <svg height="20" width="100">
                
              </svg>
            </div>
          </div>
        </div> */}
        
        {/* {this.props.checklistMode && <div style={{boxSizing: "border-box", borderBottom: "0px solid #999999", height: "30px", borderRadius: "10px 10px 10px 10px", width: "600px", marginRight: "auto", marginLeft: "auto", marginBottom: "12px", background: "#f5f5f5", padding: "8px"}}>
          <ContentEditable defaultText="template string" text={this.state.suggestionsTemplate} onInput={this.inputSuggestionsTemplate} />
        </div>} */}

        {this.props.checklistMode && <div style={{marginBottom: "12px", borderRadius: "10px 10px 10px 10px", background: "#f5f5f5"}}>
          <Row
            ref={(el) => this.suggestionsTemplateRow = el}
            id="suggestions_template_row"
            topic={this.state.topic}
            isSuggestion={false}
            hideBorder={true}
            value1Default="Input template string"
            value2Default="Output template string"
            hideButtons={true}
            comm={this.comm}
            selectWidth={maxSelectWidth}
            forceRelayout={this.debouncedForceUpdate}
            inFillin={inFillin}
            scrollParent={document.body}
            scoreColumns={this.state.score_columns}
          />  
        </div>}

        {!this.state.read_only && <div className={this.state.suggestionsDropHighlighted ? "gadfly-drop-highlighted gadfly-suggestions-box" : "gadfly-suggestions-box"} style={{paddingTop: "39px"}} onDragOver={this.onSuggestionsDragOver} onDragEnter={this.onSuggestionsDragEnter}
         onDragLeave={this.onSuggestionsDragLeave} onDrop={this.onSuggestionsDrop}>
          <div class="gadfly-scroll-wrap" style={{maxHeight: 31*this.state.max_suggestions, overflowY: "auto"}} ref={(el) => this.suggestionsScrollWrapRef = el}>
            {this.state.suggestions
                //.slice(this.state.suggestions_pos, this.state.suggestions_pos + this.state.max_suggestions)
                // .filter(id => {
                //   //console.log("Math.max(...this.comm.data[id].scores.map(x => x[1]))", Math.max(...this.comm.data[id].scores.map(x => x[1])))
                //   return this.comm.data[id] && this.comm.data[id].scores && Math.max(...this.comm.data[id].scores.map(x => x[1])) > 0.3
                // })
                .map((id, index) => {
              return <div key={id}>
                <Row
                  id={id}
                  ref={(el) => this.rows[id] = el}
                  topic={this.state.topic}
                  isSuggestion={true}
                  topicFilter={this.state.topicFilter}
                  value1Filter={this.state.value1Filter}
                  comparatorFilter={this.state.comparatorFilter}
                  value2Filter={this.state.value2Filter}
                  value1Default="New value"
                  value2Default="New value"
                  value2Edited={this.value2Edited}
                  selected={this.state.selections[id]}
                  soleSelected={this.state.selections[id] && Object.keys(this.state.selections).length == 1}
                  onSelectToggle={this.toggleSelection}
                  comm={this.comm}
                  scoreFilter={this.state.do_score_filter && this.state.suggestions.length > this.state.max_suggestions && index > this.state.max_suggestions-4 && this.state.score_filter}
                  selectWidth={maxSelectWidth}
                  forceRelayout={this.debouncedForceUpdate}
                  inFillin={inFillin}
                  scrollParent={this.suggestionsScrollWrapRef}
                  giveUpSelection={this.removeSelection}
                  scoreColumns={this.state.score_columns}
                />
              </div>
            })}
            {this.state.do_score_filter && this.state.suggestions.length > this.state.max_suggestions &&
              <div onClick={e => this.removeScoreFilter(e)} className="gadfly-row-add-button gadfly-hover-opacity" style={{lineHeight: "25px", display: "inline-block",}}>
                <FontAwesomeIcon icon={faChevronDown} style={{fontSize: "14px", color: "#000000", display: "inline-block"}} />
              </div>
            }
            <div style={{height: "15px"}}></div>
          </div>
          
          <div className="gadfly-suggestions-box-after"></div>
          <div style={{position: "absolute", top: "10px", width: "100%"}}>
            {this.state.suggestions.length > 1 &&
              <div onClick={this.clearSuggestions} className="gadfly-row-add-button gadfly-hover-opacity" style={{marginTop: "0px", left: "6px", top: "4px", lineHeight: "14px", cursor: "pointer", position: "absolute", display: "inline-block"}}>
                <FontAwesomeIcon icon={faTimes} style={{fontSize: "14px", color: "#000000", display: "inline-block"}} />
              </div>
            }
            {!this.state.disable_suggestions && 
              <div onClick={this.refreshSuggestions} style={{opacity: this.state.tests.length > 0 ? "0.6" : "0.2", cursor: this.state.tests.length > 0 ? "pointer" : "default", display: "inline-block", padding: "2px", paddingLeft: "15px", paddingRight: "15px", marginBottom: "5px", background: "rgba(221, 221, 221, 0)", borderRadius: "7px"}}>
                <div style={{width: "15px", display: "inline-block"}}><FontAwesomeIcon className={this.state.loading_suggestions ? "fa-spin" : ""} icon={faRedo} style={{fontSize: "13px", color: "#000000", display: "inline-block"}} /></div>
                <span style={{fontSize: "13px", fontWeight: "bold"}}>&nbsp;&nbsp;Suggestions</span>
                {/* {!this.props.checklistMode && <select dir="rtl" title="Current suggestion engine" className="gadfly-plain-select" onClick={e => e.stopPropagation()} value={this.state.engine} onChange={this.changeEngine} style={{position: "absolute", color: "rgb(170, 170, 170)", marginTop: "1px", right: "13px"}}>
                  <option value="davinci-msft">Creative</option>
                  <option value="davinci-instruct-beta">Creative Backup</option>
                  <option value="curie-msft">Fast</option>
                </select>} */}
              </div>
            }
            {this.state.suggestions_error && 
              <div style={{cursor: "pointer", color: "#990000", display: "block", fontWeight: "bold", padding: "2px", paddingLeft: "15px", paddingRight: "15px", marginTop: "-5px"}}>
                The suggestion server had an error and no suggestions were returned, you might try again.
              </div>
            }
            {this.state.loading_suggestions && this.state.tests.length < 5 &&
              <div style={{cursor: "pointer", color: "#995500", display: "block", fontWeight: "normal", padding: "2px", paddingLeft: "15px", paddingRight: "15px", marginTop: "-5px"}}>
                Warning: Auto-suggestions may perform poorly with less than five tests in the current topic!
              </div>
            }
          </div>
        </div>}

        {/* <div style={{textAlign: "left"}}>
          <div onClick={this.onOpen} class="gadfly-top-add-button" style={{marginLeft: "12px", lineHeight: "14px", opacity: "0.2", cursor: "pointer", paddingLeft: "4px", marginRight: "3px", paddingRight: "0px", display: "inline-block"}}>
            <FontAwesomeIcon icon={faPlus} style={{fontSize: "14px", color: "rgb(10, 10, 10)", display: "inline-block"}} />
          </div>
          <div onClick={this.onOpen} class="gadfly-top-add-button" style={{marginLeft: "10px", lineHeight: "14px", opacity: "0.2", cursor: "pointer", paddingLeft: "4px", marginRight: "3px", paddingRight: "0px", display: "inline-block"}}>
            <FontAwesomeIcon icon={faFolderPlus} style={{fontSize: "14px", color: "rgb(10, 10, 10)", display: "inline-block"}} />
          </div>
        </div> */}
        
        <div className="gadfly-children-frame">
          {this.state.tests.length == 0 && <div style={{textAlign: "center", fontStyle: "italic", padding: "10px", fontSize: "14px", color: "#999999"}}>
            This topic is empty. Click the plus (+) button to add a test.
          </div>}
          {this.state.tests.map((id, index) => {
            return <div key={id}>
              <Row
                id={id}
                ref={(el) => this.rows[id] = el}
                topic={this.state.topic}
                hideBorder={index == 0}
                topicFilter={this.state.topicFilter}
                value1Filter={this.state.value1Filter}
                comparatorFilter={this.state.comparatorFilter}
                value2Filter={this.state.value2Filter}
                value2Edited={this.value2Edited}
                selected={this.state.selections[id]}
                soleSelected={this.state.selections[id] && Object.keys(this.state.selections).length == 1}
                onOpen={this.setLocation}
                onSelectToggle={this.toggleSelection}
                onDrop={this.onDrop}
                updateTotals={(passes, failures) => {
                  this.totalPassesObj.setSubtotal(id, passes);
                  this.totalFailuresObj.setSubtotal(id, failures);
                }}
                comm={this.comm}
                selectWidth={maxSelectWidth}
                forceRelayout={this.debouncedForceUpdate}
                inFillin={inFillin}
                scrollParent={document.body}
                generateTopicName={this.generateTopicName}
                setSelected={this.setSelected}
                scoreColumns={this.state.score_columns}
              />
            </div>
          })}
        </div>
      </div>
      {this.state.experiment1 && <div style={{textAlign: "center", marginTop: "30px"}}>
        <a href="https://go.microsoft.com/?linkid=2028325">Contact Us</a> | <a href="https://go.microsoft.com/fwlink/?LinkId=521839">Privacy &amp; Cookies</a
        > | <a href="https://www.microsoft.com/en-us/legal/intellectualproperty/copyright/default.aspx">Terms of Use</a> | 
        <a href="https://go.microsoft.com/fwlink/?LinkId=506942">Trademarks</a> | © 2021 Microsoft
      </div>}
      {/* <div style={{borderRadius: "8px", background: "#77f", padding: "20px", position: "relative", zIndex: 1000}}>
          This is test documentation.
      </div> */}

        <div style={{textAlign: "right"}}>
          <div onClick={this.onOpen} class="gadfly-top-add-button" style={{marginRight: "0px", color: "rgb(26, 127, 55)", width: "50px", lineHeight: "14px", textAlign: "center", paddingLeft: "0px", paddingRight: "0px", display: "inline-block"}}>
            <FontAwesomeIcon icon={faCheck} style={{fontSize: "14px", color: "rgb(26, 127, 55)", display: "inline-block"}} /><br />
            <span style={{lineHeight: "20px"}}>{totalPasses}</span>
            {/* <span style={{lineHeight: "20px"}}>{this.state.tests.reduce((total, value) => total + this.rows[value].totalPasses["score"], 0)}</span> */}
          </div>
          <div onClick={this.onOpen} class="gadfly-top-add-button" style={{marginRight: "12px", marginLeft: "0px", color: "rgb(207, 34, 46)", width: "50px", lineHeight: "14px", textAlign: "center", paddingRight: "0px", display: "inline-block"}}>
            <FontAwesomeIcon icon={faTimes} style={{fontSize: "14px", color: "rgb(207, 34, 46)", display: "inline-block"}} /><br />
            <span style={{lineHeight: "20px"}}>{totalFailures}</span>
          </div>
        </div>
    </div>);
  }

  clickComparatorFilter(e) {
    console.log("clickComparatorFilter", e);
    // e.preventDefault();
    // e.stopPropagation();
    // defer(() => this.comparatorFilterEditable.focus());

    // if (!this.state.comparatorFilterEditing) {
    //   this.setState({comparatorFilterEditing: true});
    //   console.log("clickComparatorFilter editing", this.state.comparatorFilterEditing)
    //   e.preventDefault();
    //   e.stopPropagation();
    //   defer(() => this.value1Editable.focus());
    // }
  }

  clickModel(modelName, e) {
    if (modelName !== this.state.score_columns[0]) {
      this.comm.send(this.id, {action: "set_first_model", model: modelName});
    }
  }

  // This is a poor man's hack for what should eventually be multiple cursors
  value2Edited(id, old_value, new_value) {
    const keys = Object.keys(this.state.selections);
    console.log("value2Editedcccccc", id, old_value, new_value, keys)
    if (keys.length > 1 && this.state.selections[id]) {
      for (const k in this.state.selections) {
        console.log("K", k, this.comm.data[k].value2, id)
        if (k !== id && this.rows[k].state.value2 === old_value) {
          // console.log("setting new value", new_value)
          this.rows[k].setValue2(new_value);
        }
      }
    }
  }

  setSelected(id) {
    console.log("setSelected", id)
    let selections = {};
    selections[id] = true;
    this.setState({selections: selections});
  }

  removeSelection(id) {
    console.log("removeSelection", id)
    let newId = undefined;
    const ids = this.state.suggestions.concat(this.state.tests);
    for (let i = 0; i < ids.length; i++) {
      console.log(i, ids[i], id);
      if (ids[i] === id) {
        console.log(i, ids[i]);
        if (i+1 < ids.length) {
          console.log("i+1", i+1, ids[i+1]);
          newId = ids[i+1];
        } else if (i > 0) {
          console.log("i-1", i-1, ids[i-1]);
          newId = ids[i-1];
        }
        break;
      }
    }

    // change our selection to the new id
    if (newId !== undefined) {
      console.log(newId);
      let selections = {};
      selections[newId] = true;
      this.setState({selections: selections});
    }
  }

  changeEngine(e) {
    this.comm.send(this.id, {"engine": e.target.value});
    this.setState({engine: e.target.value});
  }

  setLocation(pathname) {
    console.log("setLocation", pathname);
    this.props.history.push(this.props.prefix + pathname);
    this.setState({selections: {}});
  }

  locationChanged(location, action) {
    console.log("locationChanged", location, action);
    this.goToTopic(this.stripPrefix(location.pathname));
  }

  newData(data) {
    if (data === undefined) return;

    if (data && "suggestions" in data && !("loading_suggestions" in data)) {
      data["loading_suggestions"] = false;
    }

    console.log("data", data);

    // always select new topics for renaming when they exist
    for (let i in data.tests) {
      if (data.tests[i].startsWith("/New topic")) {
        data.selections = {};
        data.selections[data.tests[i]] = true;
        break;
      }
    }

    // select the first suggestion if we have several and there is no current selection
    if (Object.keys(this.state.selections).length === 0 && data.suggestions.length > 1) {
      data.selections = {};
      data.selections[data.suggestions[0]] = true;
    }
    
    this.setState(data);

    // TODO: this is just a hack for the Checklist baseline user study
    defer(() => {
      //console.log("this.state.suggestions", this.state.suggestions, this.suggestionsTemplateRow.state.value2);
      // fill in the value of the template output if it is blank and we have a guess
      if (this.state.suggestions && this.suggestionsTemplateRow && (this.suggestionsTemplateRow.state.value2 === null || this.suggestionsTemplateRow.state.value2 === "")) {
        
        const key = this.state.suggestions[this.state.suggestions.length - 1];
        //console.log("key", key, Object.keys(this.comm.data))
        if (key in this.comm.data) {
          //console.log("DDFSSS")
          this.suggestionsTemplateRow.setState({value2: this.comm.data[key].value2, comparator: this.comm.data[key].comparator});
        }
      }
    });
  }

  clockFinished() {
    console.log("clockFinished")
    this.setState({
      timerExpired: true
    });
    setTimeout(() => {
      this.setState({
        timerExpired: false,
        experiment_pos: this.state.experiment_pos + 1
      });
      var loc = this.state.experiment_locations[this.state.experiment_pos].location;
      if (loc.startsWith("//")) { // double slash is used to get out of the current test tree
        window.location = loc.slice(1);
      } else if (loc.startsWith("/")) {
        this.setLocation(loc);
      } else {
        window.location = loc;
      }
    }, 5000);
  }

  openTests() {
    this.props.history.push("/Tests")
  }

  openFillins() {
    this.props.history.push("/Fill-ins")
  }

  keyDownHandler(e) {
    let newId = undefined;
    if (e.keyCode == 8 || e.keyCode == 46) { // backspace and delete
      const keys = Object.keys(this.state.selections);
      const ids = this.state.suggestions.concat(this.state.tests);
      if (keys.length > 0) {
        
        // select the next test after the selected one
        let lastId = undefined;
        for (const i in ids) {
          if (this.state.selections[lastId] !== undefined && this.state.selections[ids[i]] === undefined) {
            newId = ids[i];
            break;
          }
          lastId = ids[i];
        }
        let selections = {};//clone(this.state.selections);
        if (newId !== undefined) selections[newId] = true;
        this.setState({selections: selections});
        this.comm.send(keys, {topic: "DO_DELETE__djk39sd"});
      }
    } else if (e.keyCode == 38 || e.keyCode == 40) {
      const keys = Object.keys(this.state.selections);
      const ids = this.state.suggestions.concat(this.state.tests);
      if (keys.length == 1) {
        const currId = keys[0];
        let lastId = undefined;
        for (const i in ids) {
          if (e.keyCode == 38 && ids[i] === currId) { // up arrow
            newId = lastId;
            break; 
          }
          if (e.keyCode == 40 && currId === lastId) { // down arrow
            newId = ids[i];
            break;
          }
          lastId = ids[i];
        }
        console.log(" arrow!", lastId, currId);
      } else if (keys.length === 0) {
        if (this.state.suggestions.length > 1) {
          newId = this.state.suggestions[0];
        } else {
          newId = this.state.tests[0];
        }
      }
      console.log(" arrow!", keys, newId);
    } else {
      return;
    }
    e.preventDefault();
    e.stopPropagation();

    // change our selection to the new id
    if (newId !== undefined) {
      let selections = {};//clone(this.state.selections);
      selections[newId] = true;
      this.setState({selections: selections});
    }
  }

  
  removeScoreFilter(e) {
    e.preventDefault();
    e.stopPropagation();

    this.setState({do_score_filter: false});
  }

  generateTopicName() {

    // get a unique new topic name
    let new_topic_name = "New topic";
    let suffix = "";
    let count = 0;
    while (this.state.tests.includes(this.state.topic + "/" + new_topic_name + suffix)) {
      count += 1;
      suffix = " " + count;
    }
    new_topic_name = new_topic_name + suffix

    return new_topic_name;
  }

  addNewTopic(e) {
    e.preventDefault();
    e.stopPropagation();

    this.comm.send(this.id, {action: "add_new_topic"});
  }

  addNewTest(e) {
    e.preventDefault();
    e.stopPropagation();

    this.comm.send(this.id, {action: "add_new_test"});
  }

  inputValue2Filter(text) {
    this.setState({value2Filter: text});
  }

  inputSuggestionsTemplate(text) {
    this.setState({suggestionsTemplate: text});
  }

  inputValue1Filter(text) {
    this.setState({value1Filter: text});
  }
  
  inputComparatorFilter(text) {
    this.setState({comparatorFilter: text});
  }

  inputTopicFilter(text) {
    this.setState({topicFilter: text});
  }

  refreshSuggestions(e) {
    e.preventDefault();
    e.stopPropagation();
    console.log("refreshSuggestions");
    if (this.state.loading_suggestions || this.state.tests.length === 0) return;
    for (let k in Object.keys(this.state.selections)) {
      if (this.state.suggestions.includes(k)) {
        delete this.state.selections[k];
      }
    }
    this.setState({suggestions: [], loading_suggestions: true, suggestions_pos: 0, do_score_filter: true});
    this.comm.send(this.id, {
      action: "refresh_suggestions", value2_filter: this.state.value2Filter, value1_filter: this.state.value1Filter,
      comparator_filter: this.state.comparatorFilter,
      suggestions_template_value1: this.suggestionsTemplateRow && this.suggestionsTemplateRow.state.value1,
      suggestions_template_comparator: this.suggestionsTemplateRow && this.suggestionsTemplateRow.state.comparator,
      suggestions_template_value2: this.suggestionsTemplateRow && this.suggestionsTemplateRow.state.value2,
      checklist_mode: !!this.suggestionsTemplateRow
    });
  }

  clearSuggestions(e) {
    e.preventDefault();
    e.stopPropagation();
    console.log("clearSuggestions");
    this.setState({suggestions_pos: 0, suggestions: []});
    this.comm.send(this.id, {action: "clear_suggestions"});
  }

  pageSuggestions(e, direction) {
    e.preventDefault();
    e.stopPropagation();
    if (direction == "up") {
      this.setState({
        suggestions_pos: Math.max(0, this.state.suggestions_pos - this.state.max_suggestions)
      })
    } else {
      this.setState({
        suggestions_pos: this.state.suggestions_pos + this.state.max_suggestions
      })
    }
  }

  toggleSelection(id, shiftKey, metaKey) {
    console.log("toggleSelection", id, shiftKey, metaKey);
    
    if (!shiftKey && metaKey) {
      let selections = clone(this.state.selections);
      selections[id] = selections[id] ? false : true;
      this.setState({selections: selections});
    } else if (shiftKey) {
      const keys = Object.keys(this.state.selections);
      let first_selection_id = null;
      if (keys.length > 0) first_selection_id = keys[0];
      let selections = {};
      let selecting = false;
      console.log("first_selection_id", first_selection_id)
      for (let i = 0; i < this.state.suggestions.length; ++i) {
        const curr_id = this.state.suggestions[i];
        if (curr_id === id) {
          if (selecting) {
            selections[curr_id] = true;
            selecting = false;
          } else {
            selecting = true;
          }
        }
        if (curr_id === first_selection_id) {
          if (selecting) {
            selections[curr_id] = true;
            selecting = false;
          } else {
            selecting = true;
          }
        }
        if (selecting) {
          selections[curr_id] = true;
        }
      }
      for (let i = 0; i < this.state.tests.length; ++i) {
        const curr_id = this.state.tests[i];
        if (curr_id === id) {
          if (selecting) {
            selections[curr_id] = true;
            selecting = false;
          } else {
            selecting = true;
          }
        }
        if (curr_id === first_selection_id) {
          if (selecting) {
            selections[curr_id] = true;
            selecting = false;
          } else {
            selecting = true;
          }
        }
        if (selecting) {
          selections[curr_id] = true;
        }
      }
      
      this.setState({selections: selections});
    } else {
      const keys = Object.keys(this.state.selections);
      if (!this.state.selections[id]) {
        let selections = {};
        selections[id] = true;
        console.log("setting state from toggleSelection", keys, id)
        this.setState({selections: selections});
      }
    }
  }

  onSuggestionsDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  onSuggestionsDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    this.setState({suggestionsDropHighlighted: this.state.suggestionsDropHighlighted - 1});
  }

  onSuggestionsDragEnter(e) {
    console.log("eeenter", e.target)
    e.preventDefault();
    e.stopPropagation();
    this.setState({suggestionsDropHighlighted: this.state.suggestionsDropHighlighted + 1});
  }

  onSuggestionsDrop(e) {
    
    e.preventDefault();
    e.stopPropagation();
    const id = e.dataTransfer.getData("id");
    console.log("onSuggestionsDrop", e, id);
    if (this.state.suggestions.indexOf(id) !== -1) return; // dropping a suggestion into suggestions should do nothing
    this.setState({suggestionsDropHighlighted: 0});
    this.onDrop(id, {topic: "suggestion"});
  }

  onDrop(id, data) {
    console.log("onDrop", id, data)
    let ids;
    if (this.state.selections[id]) {
      ids = Object.keys(this.state.selections);
      this.setState({selections: {}});
    } else ids = id;
    this.comm.send(ids, data);
  }

  goToTopic(topic) {
    console.log("goToTopic", topic);
    if (this.suggestionsTemplateRow) {
      this.suggestionsTemplateRow.setState({value2: null});
    }
    this.comm.send(this.id, {action: "change_topic", topic: stripSlash(topic)});
  }
}

function stripSlash(str) {
  return str.endsWith('/') ? str.slice(0, -1) : str;
}

const IOPairChartWithRouter = withRouter(IOPairChart);

export default class Gadfly extends React.Component {

  constructor(props) {
    super(props);
    //this.updateState = this.updateState.bind(this);
    console.log("interfaceId", this.props.interfaceId)
    this.state = {enabled: true};
    //this.comm = new JupyterComm('gadfly_state_target_'+this.props.interfaceId, this.updateState);
    window.gadfly_root = this;
  }

  // updateState(state) {
  //   this.setState(state);
  // }

  render() {

    const Router = this.props.environment === "web" ? BrowserRouter : MemoryRouter;

    return (
      <div style={{maxWidth: "1000px", marginLeft: "auto", marginRight: "auto"}}>
        <div style={{paddingLeft: "0px", width: "100%", fontFamily: "Helvetica Neue, Helvetica, Arial, sans-serif", boxSizing: "border-box", fontSize: "13px", opacity: this.state.enabled ? 1 : 0.4}}>
          <Router>
            <IOPairChartWithRouter
              interfaceId={this.props.interfaceId} environment={this.props.environment}
              websocket_server={this.props.websocket_server} enabled={this.state.enabled}
              startingTopic={this.props.startingTopic} checklistMode={this.props.checklistMode}
              prefix={this.props.prefix}
            />
          </Router>
        </div>
      </div>
    );
  }
}

function getMouseEventCaretRange(evt) {
    var range, x = evt.clientX, y = evt.clientY;

    // Try the simple IE way first
    if (document.body.createTextRange) {
        range = document.body.createTextRange();
        range.moveToPoint(x, y);
    }

    else if (typeof document.createRange != "undefined") {
        // Try Mozilla's rangeOffset and rangeParent properties,
        // which are exactly what we want
        if (typeof evt.rangeParent != "undefined") {
            range = document.createRange();
            range.setStart(evt.rangeParent, evt.rangeOffset);
            range.collapse(true);
        }

        // Try the standards-based way next
        else if (document.caretPositionFromPoint) {
            var pos = document.caretPositionFromPoint(x, y);
            range = document.createRange();
            range.setStart(pos.offsetNode, pos.offset);
            range.collapse(true);
        }

        // Next, the WebKit way
        else if (document.caretRangeFromPoint) {
            range = document.caretRangeFromPoint(x, y);
        }
    }

    return range;
}

window.Gadfly = Gadfly

