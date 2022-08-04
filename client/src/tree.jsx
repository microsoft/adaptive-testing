import React from 'react';
import autoBind from 'auto-bind';
import { faPlus, faFolderPlus, faSquareList, faRectangleList, faSquare, faCircleCheck, faCircleDot, faCircle, faTimes, faChevronDown, faFolderOpen, faFolder, faChevronRight, faRedo, faFilter } from '@fortawesome/free-regular-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'

export default class Tree extends React.Component {
  constructor(props) {
    super(props);
    autoBind(this);
    
    this.state = {
      dropHighlighted: 0,
      open: this.props.open,
    };
  }

  render() {
    const open = this.state.open || this.props.selected || this.props.child_selected;
    let icon;
    if (this.props.label === "expectation_marker") {
      icon = faSquare;
    } else if (this.props.name == "Uncategorized") {
      icon = faRectangleList;
    } else if (open) {
      icon = faFolderOpen;
    } else {
      icon = faFolder;
    }
    return <div style={{marginLeft: (parseFloat(this.props.level) * 4) + "px"}}>
      <div style={{display: "flex", alignItems: "stretch"}}>
        <div style={{width: "4px", background: this.props.selected ? "rgb(9, 105, 218)" : "none", borderRadius: "2px", marginRight: "5px", display: "flex", alignItems: "center"}}>
          {/* <svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 504.35" width="14" style={{marginLeft: "-18px", opacity: this.props.selected ? 1 : 0}}>
            <path d="M64,199.35a64,64,0,0,0-64,64v177a64,64,0,0,0,64,64H416a64,64,0,0,0,64-64v-177a63.49,63.49,0,0,0-64-64m0,48a16,16,0,0,1,16,16v177a16,16,0,0,1-16,16H64a16,16,0,0,1-16-16v-177a16,16,0,0,1,16-16" style={{fill: "rgb(9, 105, 218)"}}/>
            <path d="M122.8,199.35v-47c0-75.58,44.27-128.85,119.85-128.85S364.5,76.77,364.5,152.35v47" style={{fill: "none", stroke: "rgb(9, 105, 218)", strokeMiterlimit: 10, strokeWidth: "47px"}}/>
          </svg> */}
          {/* <svg viewBox="0 0 276.99 507.01" width="8" style={{marginLeft: "-18px", opacity: this.props.selected ? 1 : 0}}>
            <circle cx="138.49" cy="138.49" r="114.99" style={{fill: "none", stroke: "rgb(9, 105, 218)", strokeMiterlimit: 10, strokeWidth: "47px"}}/>
            <line x1="138.49" y1="253.49" x2="138.49" y2="483.51" style={{fill: "none", stroke: "rgb(9, 105, 218)", strokeLinecap: "round", strokeMiterlimit: 10, strokeWidth: "47px"}}/>
          </svg> */}
          <FontAwesomeIcon onClick={this.toggleOpen} icon={faSquare} style={{fontSize: "14px", color: "#000", opacity: this.props.selected ? 0.2 : 0, display: "inline-block", marginLeft: "-20px"}} />
        </div>
        <div style={{borderRadius: "7px 7px 7px 7px", paddingLeft : "4px", flexGrow: 1, paddingRight : "4px", background: this.props.selected ? "rgba(1, 1, 1, 0.04)" : "none", cursor: "pointer"}} 
              onClick={this.onClick} onDragOver={this.onDragOver} onDragEnter={this.onDragEnter}
              onDragLeave={this.onDragLeave} onDrop={this.onDrop}>
          {this.props.show_label &&
            <div style={{padding: "6px", fontStyle: this.props.name == "Unspecified" ? "italic" : "normal"}}>
              <div style={{display: "inline-block", width: "20px", overflow: "clip"}}>
                <FontAwesomeIcon onClick={this.toggleOpen} icon={icon} style={{fontSize: "14px", color: "#666666", display: "inline-block"}} />
              </div>
              {this.props.name === "" ? "Tests" : decodeURIComponent(this.props.name)}
            </div>
          }
          {this.props.children && open && this.props.children.map((child, index) => {
            return <Tree key={index} level={parseFloat(this.props.level)} onClick={this.props.onClick} label={child.label} topic={child.topic} show_label={true} children={child.children || []} name={child.name} selected={child.selected} child_selected={child.child_selected} />;
          })}
        </div>
      </div>
      
      {this.props.selected && false && <div style={{marginLeft: ((parseFloat(this.props.level)+1) * 4 + 4 + 4 + 6) + "px", padding: "6px", paddingLeft: "7px", paddingTop: "4px"}}>
        {/* <FontAwesomeIcon onClick={this.toggleOpen} icon={faPlus} style={{fontSize: "14px", color: "#666666", display: "inline-block"}} /> */}
        {/* <svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 292.66 292.66" width="12" height="12" ><path style={{opacity: 0.4}} d="M146.33,292.66a23.5,23.5,0,0,1-23.5-23.5V169.83H23.5a23.5,23.5,0,0,1,0-47h99.33V23.5a23.5,23.5,0,1,1,47,0v99.33h99.33a23.5,23.5,0,0,1,0,47H169.83v99.33A23.5,23.5,0,0,1,146.33,292.66Z"/></svg> */}
        <svg viewBox="0 0 512 512" height="14" width="16" style={{opacity: 0.4}}>
          <path d="M448,112H256L192.81,52.69C193,43.29,70.15,49.84,64,48A16,16,0,0,0,48,64V384a16,16,0,0,0,16,16H448a16,16,0,0,0,16-16V128A16,16,0,0,0,448,112ZM345.6,282.8H286.4V342a23.5,23.5,0,0,1-47,0V282.8H180.2a23.5,23.5,0,0,1,0-47h59.2V176.6a23.5,23.5,0,0,1,47,0v59.2h59.2a23.5,23.5,0,0,1,0,47Z" style={{fill: "none"}}/>
          <path d="M448,64H275.9L227.6,18.75A63.8,63.8,0,0,0,182.4,0H64A64,64,0,0,0,0,64V384a64,64,0,0,0,64,64H448a64,64,0,0,0,64-64V128A63.49,63.49,0,0,0,448,64Zm16,320a16,16,0,0,1-16,16H64a16,16,0,0,1-16-16V64A16,16,0,0,1,64,48c6.15,1.84,129-4.71,128.81,4.69L256,112H448a16,16,0,0,1,16,16Z"/>
          <path d="M345.6,235.8H286.4V176.6a23.5,23.5,0,0,0-47,0v59.2H180.2a23.5,23.5,0,0,0,0,47h59.2V342a23.5,23.5,0,0,0,47,0V282.8h59.2a23.5,23.5,0,0,0,0-47Z"/>
        </svg>
      </div>}
    </div>
  }

  toggleOpen(e) {
    e.preventDefault();
    e.stopPropagation();
    this.setState({open: !this.state.open});
  }

  onClick(e) {
    console.log("tree clicked", this.props)
    e.preventDefault();
    e.stopPropagation();
    if (this.props.onClick) {
      this.props.onClick(this.props.topic);
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