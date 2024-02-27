import React from 'react';
import autoBind from 'auto-bind';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faCheck, faBan, faFolderMinus, faArrowRight, faTimes, faFolderPlus, faFolder} from '@fortawesome/free-solid-svg-icons'
import { defer, debounce } from 'lodash';
import { changeInput, changeLabel, changeOutput, deleteTest, moveTest, redraw } from './CommEvent';
import ContentEditable from './content-editable';
import ContextMenu from './context-menu';
import JupyterComm from './jupyter-comm';
import WebSocketComm from './web-socket-comm'
import { useSelector } from 'react-redux';
import { AppDispatch, RootState } from './store';
import { refreshBrowser } from './utils';


interface RowBaseProps {
  id: string;
  soleSelected: boolean;
  forceRelayout: () => void;
  scoreColumns?: any[];
  updateTotals?: (id: string, passes: number, failures: number) => void;
  scrollParent: HTMLElement | null;
  value1Filter?: string;
  value2Filter?: string;
  comparatorFilter?: string;
  scoreFilter?: number;
  selected: boolean;
  isSuggestion?: boolean;
  comm: JupyterComm | WebSocketComm;
  dispatch: AppDispatch;
  hideBorder?: boolean;
  outputColumnWidth: string;
  inputDefault?: string;
  outputDefault?: string;
  user: string;
  topic: string;
  giveUpSelection?: (id: string) => void;
  onOpen?: (topic: string) => void;
  onSelectToggle: (id: string, shiftKey: any, metaKey: any) => void;
  onDrop?: (id: string, topic: string) => void;
}

interface RowProps extends RowBaseProps {
  // The test / topic / suggestion data
  rowData: any;
}

interface RowState {
  type?: any;
  label?: string;
  // topic_name?: string;
  comparator: string;
  dropHighlighted: number; // used as a boolean
  dragging: boolean; // used anywhere?
  hovering: boolean;
  plusHovering: boolean;
  hidden?: boolean;
  // editing: boolean;
  value1: string;
  value2: string;
  display_parts: {};
  contextTop?: number;
  contextLeft?: number;
  contextOpen?: boolean;
  contextRows?: any[];
  description?: string;
  previewValue1?: boolean;
  previewValue2?: boolean;
  prefix?: string;
  maxImageHeight: number;
  contextFocus?: string;
}


const Row = React.forwardRef((props: RowBaseProps, ref: React.LegacyRef<RowInternal>) => {
  const testTree = useSelector((state: RootState) => state.testTree);
  let data: any = null;
  if (props.isSuggestion) {
    if (Object.keys(testTree.suggestions).includes(props.id)) {
      data = testTree.suggestions[props.id];
    } else {
      console.error("Could not find suggestions in TestTree store for id", props.id);
      return null;
    }
  } else {
    if (Object.keys(testTree.tests).includes(props.id)) {
      data = testTree.tests[props.id];
    } else {
      console.error("Could not find tests in TestTree store for id", props.id);
      return null;
    }
  }

  return <RowInternal rowData={data} ref={ref} {...props} />
});

export default Row;

export class RowInternal extends React.Component<RowProps, RowState> {
  dataLoadActions: any[];
  scrollToView: boolean;
  divRef: HTMLDivElement | null;
  topicNameEditable: ContentEditable | null;
  inputEditable: ContentEditable | null;
  outputEditable: ContentEditable | null;
  mouseDownTarget: HTMLElement;

  constructor(props) {
    super(props);
    autoBind(this);

    this.state = {
      // editing: false,
      input: "",
      output: "",
      labeler: "anonymous",
      scores: null,
      dragging: false,
      dropHighlighted: 0,
      hovering: false,
      plusHovering: false,
      maxImageHeight: 100,
      display_parts: {},
      value1: "",
      value2: "",
      comparator: "",
      ...props.data
    };

    this.dataLoadActions = [];

    window["row_"+this.props.id] = this;
    window.faTimes = faTimes;
  }

  UNSAFE_componentWillUpdate(nextProps, nextState) {

    // if we are becoming to sole selected item then we should scroll to be viewable after rendering
    if (!this.props.soleSelected && nextProps.soleSelected) {
      this.scrollToView = true;
    }

    // we need to force a relayout if the type changed since that impacts global alignments
    if (this.state.type !== nextState.type) {
      if (this.props.forceRelayout) {
        this.props.forceRelayout();
      } 
    }
  }

  componentDidUpdate() {
    this.componentDidUpdateOrMount(false);
  }

  componentDidMount() {
    this.componentDidUpdateOrMount(true);
  }
  
  componentDidUpdateOrMount(mount) {
    // update any listeners for score totals
    if (this.props.scoreColumns) {
      for (const k of this.props.scoreColumns) {
        if (this.props.rowData.scores && this.props.updateTotals) {
          // console.log("this.props.updateTotals", k, this.props.rowData.scores[k])
          this.props.updateTotals(k,
            this.props.rowData.scores[k].reduce((total, value) => total + (value[1] <= 0), 0),
            this.props.rowData.scores[k].reduce((total, value) => total + (value[1] > 0), 0)
          );
        }
      }
    }

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
    // console.log("---- render Row ----");
    if (this.props.rowData.label == null) return null; // only render if we have data
    // console.log("real render Row");

    const main_score = this.props.scoreColumns ? this.props.scoreColumns[0] : undefined;
    // console.log("rendering row", this.props)
    // apply the value1/value2/topic filters
    if (this.props.rowData.topic_name == null) {
      if (this.props.value1Filter && this.state.value1 !== "") {
        const re = RegExp(this.props.value1Filter);
        if (!re.test(this.state.value1 ?? "")) return null;
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

    } else if (this.props.value2Filter) {
      const re = RegExp(this.props.value2Filter); // TODO: rename value2Filter to reflect it's global nature
      if (!re.test(this.props.rowData.topic_name)) return null;
    }
    // console.log("real render Row2");


    // extract the raw model outputs as strings for tooltips
    // let model_output_strings = {};
    // for (const val of ["value1", "value2", "value3"]) {
    //   model_output_strings[val] = [];
    //   const val_outputs = this.state[val+"_outputs"] || [];
    //   for (const k in val_outputs) {
    //     if (val_outputs[k] && val_outputs[k].length == 1) {
    //       const d = val_outputs[k][0][1];
    //       let str = k.slice(0, -6) + " outputs for " + val + ": \n";
    //       for (const name in d) {
    //         if (name === "string") {
    //           str += d[name] + "\n";
    //         } else {
    //           if (typeof d[name] === 'string') {
    //             str += name + ": " + "|".join(d[name].split("|").map(x => "" + parseFloat(x).toFixed(3)));
    //           } else {
    //             str += name + ": " + d[name].toFixed(3) + "\n";
    //           }
    //         }
    //       }
    //       model_output_strings[val].push(str);
    //     }
    //   }
    //   model_output_strings[val] = model_output_strings[val].join("\n");
    // }
    

    let outerClasses = "adatest-row-child";
    if (this.props.selected) outerClasses += " adatest-row-selected";
    if (this.state.dropHighlighted) outerClasses += " adatest-row-drop-highlighted";
    if (this.state.dragging) outerClasses += " adatest-row-dragging";
    if (this.props.isSuggestion && this.state.plusHovering) outerClasses += " adatest-row-hover-highlighted";
    //if (this.state.hidden) outerClasses += " adatest-row-hidden";

    // let hideClasses = "adatest-row-hide-button";
    // if (this.state.hovering) hideClasses += " adatest-row-hide-hovering";
    // if (this.state.hidden) hideClasses += " adatest-row-hide-hidden";

    // let addTopicClasses = "adatest-row-hide-button";
    // if (this.state.hovering) addTopicClasses += " adatest-row-hide-hovering";

    // let editRowClasses = "adatest-row-hide-button";
    // if (this.state.hovering) editRowClasses += " adatest-row-hide-hovering";
    // if (this.props.rowData.editing) editRowClasses += " adatest-row-hide-hidden";

    // const test_type_parts = this.props.test_type_parts[this.state.type];
    
    let overall_score = {};
    if (this.props.rowData.scores) {
      for (let k in this.props.rowData.scores) {
        const arr = this.props.rowData.scores[k].filter(x => Number.isFinite(x[1])).map(x => x[1])
        overall_score[k] = arr.reduce((a, b) => a + b, 0) / arr.length;
      }
    } else {
      for (let k in overall_score) {
        overall_score[k] = NaN;
      }
    }

    // console.log("overall_score", overall_score);

    // var hack_score = overall_score[this.props.scoreColumns[0]];
    // var hack_output_flip = {
    //   "todo": "not todo",
    //   "not todo": "todo"
    // }
    // console.log("asdfa", main_score, this.state["value1_outputs"]);
    // if (this.state["value1_outputs"]) {
    //   var tmp = this.state["value1_outputs"][main_score][0][1];
    //   console.log("heresss65", this.state["value1_outputs"], Object.keys(tmp));
    //   var hack_output_name = Object.keys(tmp).reduce((a, b) => tmp[a] > tmp[b] ? a : b);
    // }
    var label_opacity = this.props.rowData.labeler === "imputed" ? 0.5 : 1;

    // get the display parts for the template instantiation with the highest score
    // const display_parts = this.state.display_parts ? this.state.display_parts[this.state.max_score_ind] : {};

    // console.log("overall_score[main_score]", overall_score[main_score], this.props.score_filter)
    if (this.props.scoreFilter && overall_score[main_score] < this.props.scoreFilter && this.props.scoreFilter > -1000) {
      //console.log("score filter ", this.state.value1, score, this.props.scoreFilter)
      return null;
    }
    // console.log("real render Row3");
    
    // compute the confidence score for the row
    let bar_width = 0;
    if (Number.isFinite(overall_score[main_score])) {
      bar_width = Math.abs(100*scale_score(overall_score[main_score]));
    }

    return <div className={outerClasses} draggable onMouseOver={this.onMouseOver} onMouseOut={this.onMouseOut} onMouseDown={this.onMouseDown}
                onDragStart={this.onDragStart} onDragEnd={this.onDragEnd} onDragOver={this.onDragOver}
                onDragEnter={this.onDragEnter} onDragLeave={this.onDragLeave} onDrop={this.onDrop} ref={(el) => this.divRef = el}
                style={this.props.hideBorder ? {} : {borderTop: "1px solid rgb(216, 222, 228)"}} tabIndex={0} onKeyDown={this.keyDownHandler}>
      <ContextMenu top={this.state.contextTop} left={this.state.contextLeft} open={this.state.contextOpen}
                    onClose={this.closeContextMenu} rows={this.state.contextRows} onClick={this.handleContextMenuClick} />
      {this.props.rowData.topic_name != null && !this.props.isSuggestion &&
        <div onClick={this.onOpen} className="adatest-row-add-button" style={{marginLeft: "6px", lineHeight: "14px", opacity: "1", cursor: "pointer", paddingLeft: "4px", marginRight: "3px", paddingRight: "0px", display: "inline-block"}}>
          <FontAwesomeIcon icon={faFolder} style={{fontSize: "14px", color: "rgb(84, 174, 255)", display: "inline-block"}} />
        </div>
      }
      {this.props.isSuggestion && this.props.rowData.topic_name != null &&
        <div onClick={this.addToCurrentTopic} className="adatest-row-add-button adatest-hover-opacity" style={{cursor: "pointer", marginRight: "3px"}} onMouseOver={this.onPlusMouseOver} onMouseOut={this.onPlusMouseOut}>
          <FontAwesomeIcon icon={faFolderPlus} style={{fontSize: "14px", color: "#000000", display: "inline-block"}} title="Add to current topic" />
        </div>
      }
      {/* {this.props.rowData.topic_name == null &&
        <svg height="20" width="50" style={{marginTop: "5px", flex: "0 0 50px", display: "inline-block", marginLeft: "8px"}}>
          <FontAwesomeIcon icon={faTimes} height="15px" y="3px" x="15px" style={{color: "rgb(0, 0, 0)", cursor: "pointer"}} textAnchor="middle" />
          <FontAwesomeIcon icon={faCheck} height="15px" y="3px" x="-15px" style={{color: "rgba(0, 0, 0, 0.05)", cursor: "pointer"}} textAnchor="middle" />
        </svg>
      } */}
      
      <div style={{padding: "0px", flex: 1}} onClick={this.clickRow} onDoubleClick={this.onOpen}>  
        {this.props.rowData.topic_name != null ? <React.Fragment>
          <div style={{display: "flex", marginTop: "7px", fontSize: "14px"}}> 
            <div className={this.state.hidden ? "adatest-row-hidden": ""} style={{flex: "1", textAlign: "left"}}>
              <ContentEditable onClick={this.clickTopicName} finishOnReturn={true} ref={el => this.topicNameEditable = el} text={decodeURIComponent(this.props.rowData.topic_name)} onInput={this.inputTopicName} onFinish={this.finishTopicName} /*editable={this.props.rowData.editing}*/ />
              <span style={{color: "#999999"}}>{this.props.rowData.description}</span>
            </div>
          </div>
          <div className="adatest-row" style={{opacity: 0.6, marginTop: "-16px", display: this.state.previewValue1 ? 'flex' : 'none'}}>
            {/* <div style={{flex: "0 0 140px", textAlign: "left"}}>
              <span style={{color: "#aaa"}}>{this.state.prefix}</span>
            </div> */}
            <div className="adatest-row-input">
              <span style={{color: "#aaa", opacity: this.state.hovering ? 1 : 0, transition: "opacity 1s"}}>{this.state.prefix}</span><span style={{color: "#aaa"}}>"</span>{this.state.previewValue1}<span style={{color: "#aaa"}}>"</span>
            </div>
            <div style={{flex: "0 0 25px", color: "#999999", textAlign: "center", overflow: "hidden", opacity: (this.state.previewValue1 ? 1 : 0)}}>
              <div style={{lineHeight: "13px", height: "16px", opacity: "1.0", verticalAlign: "middle", display: "inline-block"}}>
                <span style={{color: "#aaa"}}>should not be</span> {/* TODO: fix this for varying comparators */}
              </div>
            </div>
            <div style={{flex: "0 0 "+this.props.outputColumnWidth, textAlign: "left"}}>
            <span style={{color: "#aaa"}}>"</span>{this.state.previewValue2}<span style={{color: "#aaa"}}>"</span>
            </div>
          </div>
          
          </React.Fragment> : (
          <div className="adatest-row">
            <div className="adatest-row-input" onClick={this.clickRow}>
              <div onClick={this.clickInput} style={{display: "inline-block"}}>
                <span style={{width: "0px"}}></span>
                {/* <span onContextMenu={this.handleInputContextMenu}> */}
                  {this.props.rowData.input.startsWith("__IMAGE=") ?
                    <img src={this.props.rowData.input.substring(8)} onDoubleClick={this.toggleImageSize} style={{maxWidth: (this.state.maxImageHeight*3)+"px", maxHeight: this.state.maxImageHeight}} />
                    :
                    <ContentEditable onClick={this.clickInput} ref={el => this.inputEditable = el} text={this.props.rowData.input} onFinish={this.finishInput} /*editable={this.props.rowData.editing}*/ defaultText={this.props.inputDefault} onTemplateExpand={this.templateExpandValue1} />
                  }
                {/* </span> */}
                <span style={{width: "0px"}}></span>
              </div>
            </div>
            <div style={{flex: "0 0 25px", display: "flex", alignItems: "center", color: "#999999", justifyContent: "center", overflow: "hidden"}}>
              <FontAwesomeIcon icon={faArrowRight} style={{fontSize: "14px", color: "#999999", display: "inline-block"}} textAnchor="left" />
            </div>
            <div onClick={this.clickOutput} style={{textDecoration: this.props.rowData.label === "off_topic" ? "line-through" : "none", maxWidth: "400px", paddingTop: "5px", paddingBottom: "5px", overflowWrap: "anywhere", background: "linear-gradient(90deg, rgba(0, 0, 0, 0.0) "+bar_width+"%, rgba(255, 255, 255, 0) "+bar_width+"%)", flex: "0 0 "+this.props.outputColumnWidth, textAlign: "left", alignItems: "center", display: "flex"}}>
              <span>
                <span style={{width: "0px"}}></span>
                <span style={{opacity: Number.isFinite(overall_score[main_score]) ? 1 : 0.5}}>
                  <ContentEditable onClick={this.clickOutput} ref={el => this.outputEditable = el} text={this.props.rowData.output} onFinish={this.finishOutput} /*editable={this.props.rowData.editing}*/ defaultText={this.props.outputDefault} />
                </span>
                <span style={{width: "0px"}}></span>
              </span>
            </div>
          </div>
        )}
      </div>
      {/* <div className="adatest-row-score-text-box"> 
        {this.props.rowData.topic_name == null && !isNaN(score) && score.toFixed(3).replace(/\.?0*$/, '')}
      </div> */}
      {/* {this.props.rowData.topic_name == null &&
        <svg height="30" width="90" style={{marginTop: "0px", flex: "0 0 90px", textAling: "left", display: "inline-block", marginLeft: "8px", marginRight: "0px"}}>
          {this.props.rowData.labeler === "imputed" && this.props.rowData.label === "pass" ?
            <FontAwesomeIcon icon={faCheck} strokeWidth="50px" style={{color: "rgba(0, 0, 0, 0.05)"}} stroke={this.props.rowData.label === "pass" ? "rgb(26, 127, 55)" : "rgba(0, 0, 0, 0.05)"} height="15px" y="8px" x="-30px" textAnchor="middle" />
          :
            <FontAwesomeIcon icon={faCheck} height="17px" y="7px" x="-30px" style={{color: this.props.rowData.label === "pass" ? "rgb(26, 127, 55)" : "rgba(0, 0, 0, 0.05)", cursor: "pointer"}} textAnchor="middle" />
          }
          {this.props.rowData.labeler === "imputed" && this.props.rowData.label === "fail" ?
            <FontAwesomeIcon icon={faTimes} strokeWidth="50px" style={{color: "rgba(0, 0, 0, 0.05)"}} stroke={this.props.rowData.label === "fail" ? "rgb(207, 34, 46)" : "rgba(0, 0, 0, 0.05)"} height="15px" y="8px" x="0px" textAnchor="middle" />
          :
            <FontAwesomeIcon icon={faTimes} stroke="" height="17px" y="7px" x="0px" style={{color: this.props.rowData.label === "fail" ? "rgb(207, 34, 46,"+label_opacity+")" : "rgba(0, 0, 0, 0.05)", cursor: "pointer"}} textAnchor="middle" />
          }
          {this.props.isSuggestion ?
            <FontAwesomeIcon icon={faBan} height="17px" y="7px" x="30px" style={{color: this.props.rowData.label === "off_topic" ? "rgb(0, 0, 0)" : "rgba(0, 0, 0, 0.05)", cursor: "pointer"}} textAnchor="middle" />
          :
            <span style={{width: "31px", display: "inline-block"}}></span>
          }
          <line x1="0" y1="15" x2="30" y2="15" style={{stroke: "rgba(0, 0, 0, 0)", strokeWidth: "30", cursor: "pointer"}} onClick={this.labelAsPass}></line>
          <line x1="30" y1="15" x2="60" y2="15" style={{stroke: "rgba(0, 0, 0, 0)", strokeWidth: "30", cursor: "pointer"}} onClick={this.labelAsFail}></line>
          <line x1="60" y1="15" x2="90" y2="15" style={{stroke: "rgba(0, 0, 0, 0)", strokeWidth: "30", cursor: "pointer"}} onClick={this.labelAsOffTopic}></line>
        </svg>
      } */}
      {this.props.scoreColumns && this.props.scoreColumns.map(k => {

        let total_pass = 0;
        if (this.props.rowData.topic_name != null && this.props.rowData.scores != null) {
          total_pass = this.props.rowData.scores[k].reduce((total, value) => total + (value[1] <= 0), 0);
        }
        let total_fail = 0;
        if (this.props.rowData.topic_name != null && this.props.rowData.scores != null) {
          total_fail = this.props.rowData.scores[k].reduce((total, value) => total + (value[1] > 0), 0);
        }

        let label_opacity = isNaN(overall_score[k]) ? 0.5 : 1;

        let scaled_score = scale_score(overall_score[k]);
        
        // this.totalPasses[k] = Number.isFinite(overall_score[k]) ? this.props.rowData.scores[k].reduce((total, value) => total + (value[1] <= 0), 0) : NaN;
        // this.totalFailures[k] = this.props.rowData.scores[k].reduce((total, value) => total + (value[1] > 0), 0);
        return <div key={k} className="adatest-row-score-plot-box">
          {/* {overall_score[k] > 0 ? */}
          <svg height="30" width="150">(total_pass / (total_pass + total_fail))
            {scaled_score < 0 &&
              <g opacity="0.05">
                <line x1="100" y1="15" x2={100 + 50*scale_score(overall_score[k])} y2="15" style={{stroke: "rgb(26, 127, 55, 1.0)", strokeWidth: "25"}}></line>
                <rect x="50" y="2.5" height="25" width="50" style={{fillOpacity: 0, stroke: "rgb(26, 127, 55, 1)", strokeWidth: "1"}} />
              </g>
            }
            {scaled_score > 0 &&
              <g opacity="0.05">
                <line x1="100" y1="15" x2={100 + 50*scale_score(overall_score[k])} y2="15" style={{stroke: "rgb(207, 34, 46, 1.0)", strokeWidth: "25"}}></line>
                <rect x="100" y="2.5" height="25" width="50" style={{fillOpacity: 0, stroke: "rgb(207, 34, 46, 1)", strokeWidth: "1"}} />
              </g>
            }
            {this.props.rowData.topic_name == null &&
              <React.Fragment>
                {/* {this.props.rowData.label == "pass" &&
                  <line x1="100" y1="15" x2={100 - (100-bar_width)/2} y2="15" style={{stroke: "rgb(26, 127, 55, 0.05)", strokeWidth: "25"}}></line>
                }
                {this.props.rowData.label == "fail" &&
                  <line x1="100" y1="15" x2={100 + bar_width/2} y2="15" style={{stroke: "rgb(207, 34, 46, 0.05)", strokeWidth: "25"}}></line>
                } */}
                {this.props.rowData.labeler === "imputed" && this.props.rowData.label === "pass" ?
                  <FontAwesomeIcon icon={faCheck} height="15px" y="8px" x="0px" strokeWidth="50px" style={{color: "rgba(0, 0, 0, 0.05)"}} stroke={this.props.rowData.label === "pass" ? "rgb(26, 127, 55)" : "rgba(0, 0, 0, 0.05)"} textAnchor="middle" />
                :
                  <FontAwesomeIcon icon={faCheck} height="17px" y="7px" x="0px" style={{color: this.props.rowData.label === "pass" ? "rgb(26, 127, 55,"+label_opacity+")" : "rgba(0, 0, 0, 0.05)", cursor: "pointer"}} textAnchor="middle" />
                }
                {this.props.rowData.labeler === "imputed" && this.props.rowData.label === "fail" ?
                  <FontAwesomeIcon icon={faTimes} height="15px" y="8px" x="50px" strokeWidth="50px" style={{color: "rgba(0, 0, 0, 0.05)"}} stroke={this.props.rowData.label === "fail" ? "rgb(207, 34, 46,"+label_opacity+")" : "rgba(0, 0, 0, 0.05)"} textAnchor="middle" />
                :
                  <FontAwesomeIcon icon={faTimes} height="17px" y="7px" x="50px" style={{color: this.props.rowData.label === "fail" ? "rgb(207, 34, 46,"+label_opacity+")" : "rgba(0, 0, 0, 0.05)", cursor: "pointer"}} textAnchor="middle" />
                }
                {this.props.rowData.labeler === "imputed" && this.props.rowData.label === "off_topic" ?
                  <FontAwesomeIcon icon={faBan} height="15px" y="8px" x="-50px" strokeWidth="50px" style={{color: "rgba(0, 0, 0, 0.05)"}} stroke="rgb(207, 140, 34, 1.0)" textAnchor="middle" />
                :
                  <FontAwesomeIcon icon={faBan} height="17px" y="7px" x="-50px" style={{color: this.props.rowData.label === "off_topic" ? "rgb(207, 140, 34, 1.0)" : "rgba(0, 0, 0, 0.05)", cursor: "pointer"}} textAnchor="middle" />
                }
                <line x1="0" y1="15" x2="50" y2="15" style={{stroke: "rgba(0, 0, 0, 0)", strokeWidth: "30", cursor: "pointer"}} onClick={this.labelAsOffTopic}></line>
                <line x1="50" y1="15" x2="100" y2="15" style={{stroke: "rgba(0, 0, 0, 0)", strokeWidth: "30", cursor: "pointer"}} onClick={this.labelAsPass}></line>
                <line x1="100" y1="15" x2="150" y2="15" style={{stroke: "rgba(0, 0, 0, 0)", strokeWidth: "30", cursor: "pointer"}} onClick={this.labelAsFail}></line>
              </React.Fragment>
            }
            {this.props.rowData.topic_name != null && total_pass > 0 &&
              <text x="75" y="16" dominantBaseline="middle" textAnchor="middle" style={{pointerEvents: "none", fill: "rgb(26, 127, 55)", fontWeight: "bold", fontSize: "14px"}}>{total_pass}</text>
            }
            {this.props.rowData.topic_name != null && total_fail > 0 &&
              <text x="125" y="16" dominantBaseline="middle" textAnchor="middle" style={{pointerEvents: "none", fill: "rgb(207, 34, 46)", fontWeight: "bold", fontSize: "14px"}}>{total_fail}</text>
            }
          </svg>
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
        // TODO: Still used?
        this.props.comm.send(this.props.id, {"action": "template_expand_value1"});
      }
    }
    this.setState({contextOpen: false});
  }

  closeContextMenu() {
    this.setState({contextOpen: false});
  }

  handleInputContextMenu(e) {
    e.preventDefault();
    console.log("handleInputContextMenu open", e, e.pageY, e.pageX)
    this.setState({contextTop: e.pageY, contextLeft: e.pageX, contextOpen: true, contextFocus: "value1", contextRows: ["Expand into a template"]});
  }

  templateExpandValue1() {
    console.log("templateExpandValue1")
    // TODO: Still used?
    this.props.comm.send(this.props.id, {"action": "template_expand_value1"});
  }

  toggleImageSize() {
    this.setState({maxImageHeight: this.state.maxImageHeight === 100 ? 500 : 100});
  }

  keyDownHandler(e) {
    if (e.keyCode == 13) {
      console.log("return!", this.props.soleSelected, this.props.selected)
      if (this.props.soleSelected) {
        if (this.props.rowData.topic_name != null) {
          this.onOpen(e);
        } else if (this.props.isSuggestion) {
          this.addToCurrentTopic(e);
          this.doOnNextDataLoad(() => {
            if (this.props.giveUpSelection != null) {
              this.props.giveUpSelection(this.props.id)
            }
          });
        }
      }
    }
  }

  doOnNextDataLoad(f) {
    this.dataLoadActions.push(f);
  }

  changeTestType(e) {
    // Still used?
    this.props.comm.send(this.props.id, {"type": e.target.value});
    this.setState({type: e.target.value});
  }

  labelAsFail(e) {
    this.setLabel("fail");
  }

  labelAsOffTopic(e) {
    this.setLabel("off_topic");
  }

  labelAsPass(e) {
    this.setLabel("pass");
  }

  setLabel(label) {
    this.props.comm.sendEvent(changeLabel(this.props.id, label, this.props.user)).then(async () => {
      if (this.props.isSuggestion) {
        await this.props.comm.sendEvent(moveTest(this.props.id, this.props.topic));
      }
      refreshBrowser(this.props.comm, this.props.dispatch);
    });
    this.setState({label: label});
  }

  // onScoreOver(e, key) {
  //   this.setState({
  //     previewValue1: this.props.comm.data[key].value1,
  //     previewValue2: this.props.comm.data[key].value2
  //   })
  // }
  // onScoreOut(e, key) {
  //   this.setState({
  //     previewValue1: null,
  //     previewValue2: null
  //   })
  // }

  // toggleEditRow(e) {
  //   e.preventDefault();
  //   e.stopPropagation();

  //   if (!this.props.rowData.editing) {
  //     this.setState({editing: true});
  //     console.log("about to edit focus")
  //     if (this.props.rowData.topic_name == null) {
  //       defer(() => this.inputEditable?.focus());
  //     } else {
  //       defer(() => this.topicNameEditable?.focus());
  //     }
  //   } else {
  //     this.setState({editing: false});
  //   }
  // }

  toggleHideTopic(e) {
    // Still used?
    e.preventDefault();
    e.stopPropagation();

    this.props.comm.send(this.props.id, {"hidden": !this.state.hidden});
  }

  /* addNewTopic(e) {
    e.preventDefault();
    e.stopPropagation();
    const newName = this.props.generateTopicName();
    const newTopic = this.props.topic + "/" + newName;
    if (this.props.rowData.topic_name == null) {
      this.props.comm.send(this.props.id, {"topic": newTopic });
    } else {
      this.props.comm.send(this.props.id, {"topic": newTopic + "/" + this.props.rowData.topic_name});
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
    console.log("this.props.rowData.topic_name XXXXXXXXXXXX", this.props.rowData.topic_name)//, "Row.onOpen(", e, ")");
    if (this.props.rowData.topic_name != null && this.props.onOpen) {
      this.props.onOpen(this.props.topic + "/" + this.props.rowData.topic_name);
    }
  }

  // inputInput(text) {
  //   console.log("inputInput", text)
  //   this.setState({input: text, scores: null});
  //   debounce(() => this.props.comm.sendEvent(changeInput(this.props.id, text))
  //     .then(() => refreshBrowser(this.props.comm, this.props.dispatch)), 500);
  // }

  finishInput(text) {
    console.log("finishInput", text)
    // this.setState({editing: false});
    if (this.props.rowData.input !== text) {
      this.props.comm.sendEvent(changeInput(this.props.id, text))
          .then(() => refreshBrowser(this.props.comm, this.props.dispatch))
    } else {
      console.log("finishInput skipping update because text is the same")
    }
    // if (text.includes("/")) {
    //   this.setState({input: text, scores: null});
    //   this.props.comm.send(this.props.id, {input: text});
    // }
  }

  finishOutput(text) {
    console.log("finishOutput", text)
    // this.setState({editing: false});
    if (this.props.rowData.output !== text) {
      this.props.comm.sendEvent(changeOutput(this.props.id, text))
            .then(() => refreshBrowser(this.props.comm, this.props.dispatch))
    } else {
      console.log("finishOutput skipping update because text is the same")
    }
    // if (text.includes("/")) {
    //   this.setState({input: text, scores: null});
    //   this.props.comm.send(this.props.id, {input: text});
    // }
  }

  // inputOutput(text) {
  //   // return;
  //   console.log("inputOutput", text);
  //   // text = text.trim(); // SML: causes the cursor to jump when editing because the text is updated
  //   debounce(() => this.props.comm.sendEvent(changeOutput(this.props.id, text))
  //     .then(() => refreshBrowser(this.props.comm, this.props.dispatch)), 500);

  //   // if (this.props.value2Edited) {
  //   //   this.props.value2Edited(this.props.id, this.state.value2, text);
  //   // }
  //   // this.setValue2(text);
  // }

  inputTopicName(text) {
    // text = text.replaceAll("\\", "").replaceAll("\n", "");
    // const encodedText = encodeURIComponent(text);
    // this.setState({topic_name: encodedText});
  }

  finishTopicName(text) {
    console.log("finishTopicName", text)
    text = encodeURIComponent(text.replaceAll("\\", "").replaceAll("\n", ""));
    let topic = this.props.topic;
    if (this.props.rowData.topic_name !== text) {
      if (this.props.isSuggestion) topic += "/__suggestions__";
      topic = topic + "/" + text;
      this.props.comm.sendEvent(moveTest(this.props.id, topic))
        .then(() => refreshBrowser(this.props.comm, this.props.dispatch));
      // this.setState({topic_name: text, editing: false});
    } else {
      console.log("finishTopicName skipping update because text is the same")
    }
  }
  
  clickRow(e) {
    if (this.props.onSelectToggle) {
      e.preventDefault();
      e.stopPropagation();
      this.props.onSelectToggle(this.props.id, e.shiftKey, e.metaKey || e.ctrlKey);
    }
  }

  clickTopicName(e) {
    console.log("clickTopicName");
    const modKey = e.metaKey || e.shiftKey || e.ctrlKey;
    if (modKey && this.props.onSelectToggle) {
      e.preventDefault();
      e.stopPropagation();
      this.props.onSelectToggle(this.props.id, e.shiftKey, e.metaKey || e.ctrlKey);
    }
    // if (!modKey && !this.props.rowData.editing) {
    //   this.setState({editing: true});
    //   console.log("topic editing", this.props.rowData.editing)
    //   e.preventDefault();
    //   e.stopPropagation();
    //   defer(() => this.topicNameEditable?.focus());
    // }
  }

  clickInput(e) {
    console.log("clickInput", e);
    const modKey = e.metaKey || e.shiftKey || e.ctrlKey;
    if (modKey && this.props.onSelectToggle) {
      e.preventDefault();
      e.stopPropagation();
      this.props.onSelectToggle(this.props.id, e.shiftKey, e.metaKey || e.ctrlKey);
    }
    // if (!modKey && !this.props.rowData.editing) {
    //   this.setState({editing: true});
    //   console.log("value1 editing", this.props.rowData.editing)
    //   e.preventDefault();
    //   e.stopPropagation();
    //   defer(() => this.inputEditable?.focus());
    // }
  }

  

  clickOutput(e) {
    console.log("clickOutput");
    const modKey = e.metaKey || e.shiftKey || e.ctrlKey;
    if (modKey && this.props.onSelectToggle) {
      e.preventDefault();
      e.stopPropagation();
      this.props.onSelectToggle(this.props.id, e.shiftKey, e.metaKey || e.ctrlKey);
    }
    // if (!modKey && !this.props.rowData.editing) {
    //   this.setState({editing: true});
    //   e.preventDefault();
    //   e.stopPropagation();
    //   defer(() => this.outputEditable?.focus());
    // }
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
    e.dataTransfer.setData("topic_name", this.props.rowData.topic_name);
    // if (this.props.onDragStart) {
    //   this.props.onDragStart(e, this);
    // }
  }

  onDragEnd(e) {
    this.setState({dragging: false});
    // if (this.props.onDragEnd) {
    //   this.props.onDragEnd(e, this);
    // }
  }

  onDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  onDragEnter(e) {
    console.log("enter", e.target)
    e.preventDefault();
    e.stopPropagation();
    if (this.props.rowData.topic_name != null) {
      this.setState({dropHighlighted: this.state.dropHighlighted + 1});
    }
  }

  onDragLeave(e) {
    console.log("leave", e.target)
    e.preventDefault();
    e.stopPropagation();
    if (this.props.rowData.topic_name != null) {
      this.setState({dropHighlighted: this.state.dropHighlighted - 1});
    }
  }

  onDrop(e) {
    
    const id = e.dataTransfer.getData("id");
    const topic_name = e.dataTransfer.getData("topic_name");
    if (this.props.rowData.topic_name != null) {
      this.setState({dropHighlighted: 0});
      if (this.props.onDrop && id !== this.props.id) {
        if (topic_name != null && topic_name !== "null" && topic_name !== "undefined") {
          this.props.onDrop(id, this.props.topic + "/" + this.props.rowData.topic_name + "/" + topic_name);
        } else {
          this.props.onDrop(id, this.props.topic + "/" + this.props.rowData.topic_name);
        }
      }
    }
  }

  addToCurrentTopic(e) {
    e.preventDefault();
    e.stopPropagation();
    console.log("addToCurrentTopic X", this.props.topic, this.props.rowData.topic_name);
    let targetTopic = this.props.topic + (this.props.rowData.topic_name == null ? "" : "/" + this.props.rowData.topic_name);
    this.props.comm.sendEvent(moveTest(this.props.id, targetTopic))
      .then(() => refreshBrowser(this.props.comm, this.props.dispatch));
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
          var scrollTop: number = childRect.top - parentRect.top;
          var scrollBot = childRect.bottom - parentViewableArea.height - parentRect.top;
        } else {
          var scrollTop: number = childRect.top;
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

// const score_min = -1;
// const score_max = 1;
function scale_score(score) {
  return score; //Math.max(Math.min(score, score_max), score_min) ///(score_max - score_min)
}