import React from 'react';
import autoBind from 'auto-bind';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faCheck, faBan, faFolderMinus, faArrowRight, faTimes, faFolderPlus, faFolder} from '@fortawesome/free-solid-svg-icons'
import { defer } from 'lodash';
import ContentEditable from './content-editable';
import ContextMenu from './context-menu';

const suggestSub = "/__suggestions__";

export default class TopicSuggestion extends React.Component {
  constructor(props) {
    super(props);
    autoBind(this);
    this.suggestedTopic = decodeURIComponent(this.props.topicId.substr(this.props.topicId.lastIndexOf("/")+1));
  }

  // getTopicName() {
  //   return decodeURIComponent(this.props.id.replace("__suggestions__/", ""));
  // }

  addToCurrentTopic(e) {
    e.preventDefault();
    e.stopPropagation();
    const newTopic = this.props.topic + "/" + this.suggestedTopic;
    console.log("addToCurrentTopic", newTopic);
    this.props.comm.send(this.props.topicId, {topic: newTopic});
  }

  render() {
    return (
      <div style={{display: "flex", marginBottom: "5px"}}>
        <div onClick={(e) => this.addToCurrentTopic(e)} style={{opacity: "1", cursor: "pointer", marginRight: "5px"}}>
          <FontAwesomeIcon icon={faFolder} style={{fontSize: "12px", color: "rgb(84, 174, 255)"}} />
        </div>
        <span>{this.suggestedTopic}</span>
      </div>
    )
  }
}