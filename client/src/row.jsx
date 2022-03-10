import React from 'react';
import autoBind from 'auto-bind';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faFolderPlus, faFolder} from '@fortawesome/free-solid-svg-icons'
import { defer } from 'lodash';
import ContentEditable from './content-editable';
import ContextMenu from './context-menu';

export default class Row extends React.Component {
  constructor(props) {
    super(props);
    autoBind(this);

    this.state = {
      editing: false,
      type: null,
      value1: null,
      value2: null,
      value3: null,
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
    // console.log("state.topic_name", state.topic_name)
    // we automatically start editing topics that are selected and have an imputed name
    if (state.topic_name && (state.topic_name.startsWith("New topic") || state.value1 === "New test") && this.props.soleSelected) {
      state["editing"] = true;
      console.log("setting editing state to true!")
    }
    
    this.setState(state);
  }

  UNSAFE_componentWillUpdate(nextProps, nextState) {

    // if we are becoming to sole selected item then we should scroll to be viewable after rendering
    if (!this.props.soleSelected && nextProps.soleSelected) {
      this.scrollToView = true;
    }

    // we need to force a relayout if the type changed since that impacts global alignments
    if (this.state.type !== nextState.type) {
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
    if (this.state.type === null) return null; // only render if we have data

    const main_score = this.props.scoreColumns ? this.props.scoreColumns[0] : undefined;
    // console.log("rendering row", this.props)
    // apply the value1/value2/topic filters
    let value1_outputs_str = "";
    let value2_outputs_str = "";
    let value3_outputs_str = "";
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

    

    let outerClasses = "adatest-row-child";
    if (this.props.selected) outerClasses += " adatest-row-selected";
    if (this.state.dropHighlighted) outerClasses += " adatest-row-drop-highlighted";
    if (this.state.dragging) outerClasses += " adatest-row-dragging";
    if (this.props.isSuggestion && this.state.plusHovering) outerClasses += " adatest-row-hover-highlighted";
    //if (this.state.hidden) outerClasses += " adatest-row-hidden";

    let hideClasses = "adatest-row-hide-button";
    if (this.state.hovering) hideClasses += " adatest-row-hide-hovering";
    if (this.state.hidden) hideClasses += " adatest-row-hide-hidden";

    let addTopicClasses = "adatest-row-hide-button";
    if (this.state.hovering) addTopicClasses += " adatest-row-hide-hovering";

    let editRowClasses = "adatest-row-hide-button";
    if (this.state.hovering) editRowClasses += " adatest-row-hide-hovering";
    if (this.state.editing) editRowClasses += " adatest-row-hide-hidden";

    const test_type_parts = this.props.test_type_parts[this.state.type];
    
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
    // console.log("overall_score[main_score]", overall_score[main_score], this.props.score_filter)
    if (this.props.scoreFilter && overall_score[main_score] < this.props.scoreFilter && this.props.scoreFiler > -1000) {
      //console.log("score filter ", this.state.value1, score, this.props.scoreFilter)
      return null;
    }

    return <div className={outerClasses} draggable onMouseOver={this.onMouseOver} onMouseOut={this.onMouseOut} onMouseDown={this.onMouseDown}
                onDragStart={this.onDragStart} onDragEnd={this.onDragEnd} onDragOver={this.onDragOver}
                onDragEnter={this.onDragEnter} onDragLeave={this.onDragLeave} onDrop={this.onDrop} ref={(el) => this.divRef = el}
                style={this.props.hideBorder ? {} : {borderTop: "1px solid rgb(216, 222, 228)"}} tabIndex="0" onKeyDown={this.keyDownHandler}>
      <ContextMenu top={this.state.contextTop} left={this.state.contextLeft} open={this.state.contextOpen}
                    onClose={this.closeContextMenu} rows={this.state.contextRows} onClick={this.handleContextMenuClick} />
      {this.state.topic_name !== null && !this.props.isSuggestion &&
        <div onClick={this.onOpen} className="adatest-row-add-button" style={{marginLeft: "6px", lineHeight: "14px", opacity: "1", cursor: "pointer", paddingLeft: "4px", marginRight: "3px", paddingRight: "0px", display: "inline-block"}}>
          <FontAwesomeIcon icon={faFolder} style={{fontSize: "14px", color: "rgb(84, 174, 255)", display: "inline-block"}} />
        </div>
      }
      {this.props.isSuggestion &&
        <div onClick={this.addToCurrentTopic} className="adatest-row-add-button adatest-hover-opacity" style={{cursor: "pointer"}} onMouseOver={this.onPlusMouseOver} onMouseOut={this.onPlusMouseOut}>
          <FontAwesomeIcon icon={this.state.topic_name === null ? faPlus : faFolderPlus} style={{fontSize: "14px", color: "#000000", display: "inline-block"}} title="Add to current topic" />
        </div>
      }
      <div style={{padding: "5px", flex: 1}} onClick={this.clickRow} onDoubleClick={this.onOpen}>  
        {this.state.topic_name !== null ? <React.Fragment>
          <div style={{display: "flex", marginTop: "3px", fontSize: "14px"}}> 
            <div className={this.state.hidden ? "adatest-row-hidden": ""} style={{flex: "1", textAlign: "left"}}>
              <ContentEditable onClick={this.clickTopicName} ref={el => this.topicNameEditable = el} text={this.state.topic_name} onInput={this.inputTopicName} onFinish={this.finishTopicName} editable={this.state.editing} />
            </div>
          </div>
          <div className="adatest-row" style={{opacity: 0.6, marginTop: "-16px", display: this.state.previewValue1 ? 'flex' : 'none'}}>
            {/* <div style={{flex: "0 0 140px", textAlign: "left"}}>
              <span style={{color: "#aaa"}}>{this.state.prefix}</span>
            </div> */}
            <div className="adatest-row-input">
              <span style={{color: "#aaa", opacity: this.state.hovering ? 1 : 0, transition: "opacity 1s"}}>{this.state.prefix}</span><span style={{color: "#aaa"}}>"</span>{this.state.previewValue1}<span style={{color: "#aaa"}}>"</span>
            </div>
            <div style={{flex: "0 0 "+this.props.selectWidth+"px", color: "#999999", textAlign: "center", overflow: "hidden", opacity: (this.state.previewValue1 ? 1 : 0)}}>
              <div style={{lineHeight: "13px", height: "16px", opacity: "1.0", verticalAlign: "middle", display: "inline-block"}}>
                <span style={{color: "#aaa"}}>should not be</span> {/* TODO: fix this for varying comparators */}
              </div>
            </div>
            <div style={{flex: "0 0 200px", textAlign: "left"}}>
            <span style={{color: "#aaa"}}>"</span>{this.state.previewValue2}<span style={{color: "#aaa"}}>"</span>
            </div>
          </div>
          
          </React.Fragment> : (
          <div className="adatest-row">
            <div className="adatest-row-input" onClick={this.clickRow}>
              <span style={{color: "#aaa"}}>{test_type_parts.text1}</span>
              &nbsp;<div onClick={this.clickValue1} style={{display: "inline-block"}}><span style={{color: "#aaa"}}>"</span><span title={value1_outputs_str} onContextMenu={this.handleValue1ContextMenu}><ContentEditable onClick={this.clickValue1} onTemplateExpand={this.templateExpandValue1} ref={el => this.value1Editable = el} text={this.state.value1} onInput={this.inputValue1} onFinish={this.finishValue1} editable={this.state.editing} defaultText={this.props.value1Default} /></span><span style={{color: "#aaa"}}>"</span></div>
            </div>
            <div style={{flex: "0 0 "+this.props.selectWidth+"px", color: "#999999", textAlign: "center", overflow: "hidden", display: "flex"}}>
              <div style={{alignSelf: "flex-end", display: "inline-block"}}>
                &nbsp;<span style={{color: "#aaa"}}><select className="adatest-plain-select" style={{marginLeft: "0px", color: "#aaa"}} value={this.state.type} onChange={this.changeTestType}>
                  {(this.props.test_types || []).map((type) => {
                    return <option key={type} value={type}>{this.props.test_type_parts[type].text2}</option>
                  })}
                </select></span>
              </div>
            </div>
            <div onClick={this.clickValue2} style={{maxWidth: "400px", overflowWrap: "anywhere", flex: "0 0 200px", textAlign: "left", display: "flex"}}>
              <span style={{alignSelf: "flex-end"}}>
                {test_type_parts.value2 === "[]" &&
                  <React.Fragment>
                    <span style={{color: "#aaa"}}>"</span><span style={{color: "#666666"}}>{this.state.value2}</span><span style={{color: "#aaa"}}>"</span><span style={{color: "#aaa", opacity: this.state.hovering ? 1 : 0, transition: "opacity 1s"}}>&nbsp;{!this.props.inFillin && "is the inversion."}</span>
                  </React.Fragment>
                }
                {test_type_parts.value2 === "{}" &&
                  <React.Fragment>
                    <span style={{color: "#aaa"}}>"</span><span title={value2_outputs_str}><ContentEditable ref={el => this.value2Editable = el} onClick={this.clickValue2} text={this.state.value2} onInput={this.inputValue2} onFinish={_ => this.setState({editing: false})} editable={this.state.editing} defaultText={this.props.value2Default} /></span><span style={{color: "#aaa"}}>"</span>
                  </React.Fragment>
                }
                {test_type_parts.text3}
                {test_type_parts.value3 === "[]" &&
                  <React.Fragment>
                    <span style={{color: "#aaa"}}>"</span><span style={{color: "#666666"}}>{this.state.value3}</span><span style={{color: "#aaa"}}>"</span><span style={{color: "#aaa", opacity: this.state.hovering ? 1 : 0, transition: "opacity 1s"}}>&nbsp;{!this.props.inFillin && "is the inversion."}</span>
                  </React.Fragment>
                }
                {test_type_parts.value3 === "{}" &&
                  <React.Fragment>
                    <span style={{color: "#aaa"}}>"</span><span title={value3_outputs_str}><ContentEditable ref={el => this.value3Editable = el} onClick={this.clickValue3} text={this.state.value3} onInput={this.inputValue3} onFinish={_ => this.setState({editing: false})} editable={this.state.editing} defaultText={this.props.value3Default} /></span><span style={{color: "#aaa"}}>"</span>
                  </React.Fragment>
                }
                {test_type_parts.text4}
              </span>
            </div>
          </div>
        )}
      </div>
      {/* <div className="adatest-row-score-text-box">
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
        return <div key={k} className="adatest-row-score-plot-box">
          {overall_score[k] > 0 ?
            <svg height="20" width="100">
              {Number.isFinite(overall_score[k]) && <React.Fragment>
                <line x1="50" y1="10" x2={50 + 48*scale_score(overall_score[k])} y2="10" style={{stroke: "rgba(0, 0, 0, 0.1)", strokeWidth: "20"}}></line>
                {this.state.scores[k].filter(x => Number.isFinite(x[1])).map((score, index) => {
                  //console.log("scale_score(score[1])", scale_score(score[1]))
                  return <line key={index} onMouseOver={e => this.onScoreOver(e, score[0])}
                              onMouseOut={e => this.onScoreOut(e, score[0])}
                              x1={50 + 48*scale_score(score[1])} y1="0"
                              x2={50 + 48*scale_score(score[1])} y2="20"
                              style={{stroke: score[1] <= 0 ? "rgb(26, 127, 55)" : "rgb(207, 34, 46)", strokeWidth: "2"}}
                        ></line>
                })}
                <line x1={50} y1="0"
                      x2={50} y2="20" strokeDasharray="2"
                      style={{stroke: "#bbbbbb", strokeWidth: "1"}}
                ></line>
                {this.state.topic_name !== null && 
                  <text x="25" y="11" dominantBaseline="middle" textAnchor="middle" style={{transition: "fill-opacity 1s, stroke-opacity 1s", strokeOpacity: this.state.hovering*1, fillOpacity: this.state.hovering*1, pointerEvents: "none", fill: "#ffffff", fontSize: "11px", strokeWidth: "3px", stroke: "rgb(26, 127, 55)", opacity: 1, strokeLinecap: "butt", strokeLinejoin: "miter", paintOrder: "stroke fill"}}>{this.state.scores[k].reduce((total, value) => total + (value[1] <= 0), 0)}</text>
                }
                {this.state.topic_name !== null &&
                  <text x="75" y="11" dominantBaseline="middle" textAnchor="middle" style={{transition: "fill-opacity 1s, stroke-opacity 1s", strokeOpacity: this.state.hovering*1, fillOpacity: this.state.hovering*1, pointerEvents: "none", fill: "#ffffff", fontSize: "11px", strokeWidth: "3px", stroke: "rgb(207, 34, 46)", opacity: 1, strokeLinecap: "butt", strokeLinejoin: "miter", paintOrder: "stroke fill"}}>{this.state.scores[k].reduce((total, value) => total + (value[1] > 0), 0)}</text>
                }
                {this.state.topic_name === null &&
                  <text x="75" y="11" dominantBaseline="middle" textAnchor="middle" style={{transition: "fill-opacity 1s, stroke-opacity 1s", strokeOpacity: this.state.hovering*1, fillOpacity: this.state.hovering*1, pointerEvents: "none", fill: "#ffffff", fontSize: "11px", strokeWidth: "3px", stroke: "rgb(207, 34, 46)", opacity: 1, strokeLinecap: "butt", strokeLinejoin: "miter", paintOrder: "stroke fill"}}>{overall_score[k].toFixed(3).replace(/\.?0*$/, '')}</text>
                }
                {this.state.topic_name === 3324 && !isNaN(overall_score[k]) &&
                  <text x={(48*scale_score(overall_score[k]) > 3000 ? 50 + 5 : 50 + 48*scale_score(overall_score[k]) + 5)} y="11" dominantBaseline="middle" textAnchor="start" style={{pointerEvents: "none", fontSize: "11px", opacity: 0.7, fill: "rgb(207, 34, 46)"}}>{overall_score[k].toFixed(3).replace(/\.?0*$/, '')}</text>
                }
              </React.Fragment>}
            </svg>
          :
            <svg height="20" width="100">
              {Number.isFinite(overall_score[k]) && <React.Fragment>
                <line x2="50" y1="10" x1={50 + 48*scale_score(overall_score[k])} y2="10" style={{stroke: "rgba(0, 0, 0, 0.1)", strokeWidth: "20"}}></line>
                {this.state.scores[k].filter(x => Number.isFinite(x[1])).map((score, index) => {
                  return <line key={index} onMouseOver={e => this.onScoreOver(e, score[0])}
                              onMouseOut={e => this.onScoreOut(e, score[0])}
                              x1={50 + 48*scale_score(score[1])} y1="0"
                              x2={50 + 48*scale_score(score[1])} y2="20"
                              style={{stroke: score[1] <= 0 ? "rgb(26, 127, 55)" : "rgb(207, 34, 46)", strokeWidth: "2"}}
                        ></line>
                })}
                <line x1={50} y1="0"
                      x2={50} y2="20" strokeDasharray="2"
                      style={{stroke: "#bbbbbb", strokeWidth: "1"}}
                ></line>
                {this.state.topic_name !== null &&
                  <text x="25" y="11" dominantBaseline="middle" textAnchor="middle" style={{transition: "fill-opacity 1s, stroke-opacity 1s", strokeOpacity: this.state.hovering*1, fillOpacity: this.state.hovering*1, pointerEvents: "none", fill: "#ffffff", fontSize: "11px", strokeWidth: "3px", stroke: "rgb(26, 127, 55)", opacity: 1, strokeLinecap: "butt", strokeLinejoin: "miter", paintOrder: "stroke fill"}}>{this.state.scores[k].reduce((total, value) => total + (value[1] <= 0), 0)}</text>
                }
                {this.state.topic_name !== null &&
                  <text x="75" y="11" dominantBaseline="middle" textAnchor="middle" style={{transition: "fill-opacity 1s, stroke-opacity 1s", strokeOpacity: this.state.hovering*1, fillOpacity: this.state.hovering*1, pointerEvents: "none", fill: "#ffffff", fontSize: "11px", strokeWidth: "3px", stroke: "rgb(207, 34, 46)", opacity: 1, strokeLinecap: "butt", strokeLinejoin: "miter", paintOrder: "stroke fill"}}>{this.state.scores[k].reduce((total, value) => total + (value[1] > 0), 0)}</text>
                }
                {this.state.topic_name === null &&
                  <text x="25" y="11" dominantBaseline="middle" textAnchor="middle" style={{transition: "fill-opacity 1s, stroke-opacity 1s", strokeOpacity: this.state.hovering*1, fillOpacity: this.state.hovering*1, pointerEvents: "none", fill: "#ffffff", fontSize: "11px", strokeWidth: "3px", stroke: "rgb(26, 127, 55)", opacity: 1, strokeLinecap: "butt", strokeLinejoin: "miter", paintOrder: "stroke fill"}}>{overall_score[k].toFixed(3).replace(/\.?0*$/, '')}</text>
                }
                {this.state.topic_name === 2342 && !isNaN(overall_score[k]) &&
                  <text x={(48*scale_score(overall_score[k]) < -3000 ? 50 - 5 : 50 + 48*scale_score(overall_score[k]) - 5)} y="11" dominantBaseline="middle" textAnchor="end" style={{pointerEvents: "none", fontSize: "11px", opacity: 0.7, fill: "rgb(26, 127, 55)"}}>{overall_score[k].toFixed(3).replace(/\.?0*$/, '')}</text>
                }
              </React.Fragment>}
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

  changeTestType(e) {
    this.props.comm.send(this.props.id, {"type": e.target.value});
    this.setState({type: e.target.value});
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
    this.setState({value2: text, scores: null});
    this.props.comm.debouncedSend500(this.props.id, {value2: text});

    // if (this.props.value2Edited) {
    //   this.props.value2Edited(this.props.id, this.state.value2, text);
    // }
    // this.setValue2(text);
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
    if (this.state.topic_name !== null) {
      this.props.comm.send(this.props.id, {topic: this.props.topic + "/" + this.state.topic_name});
    } else {
      this.props.comm.send(this.props.id, {topic: this.props.topic});
    }
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

const score_min = -1;
const score_max = 1;
function scale_score(score) {
  return Math.max(Math.min(score, score_max), score_min) ///(score_max - score_min)
}