import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faPlus, faCheck, faBan, faFolderMinus, faArrowRight, faTimes, faFolderPlus, faFolder} from '@fortawesome/free-solid-svg-icons'

export default function TopicSuggestion(props) {
  const suggestedTopic = decodeURIComponent(props.topicId.substr(props.topicId.lastIndexOf("/")+1));

  function addToCurrentTopic(e) {
    e.preventDefault();
    e.stopPropagation();
    const newTopic = props.topic + "/" + suggestedTopic;
    console.log("addToCurrentTopic", newTopic);
    props.comm.send(props.topicId, {topic: newTopic});
  }

  return (
    <div style={{display: "flex", marginBottom: "5px"}}>
      <div onClick={(e) => addToCurrentTopic(e)} style={{opacity: "1", cursor: "pointer", marginRight: "5px"}}>
        <FontAwesomeIcon icon={faFolderPlus} style={{fontSize: "12px", color: "rgb(84, 174, 255)"}} />
      </div>
      <span>{suggestedTopic}</span>
    </div>
  )
}