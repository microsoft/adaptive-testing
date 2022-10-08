import React, { useState, useRef } from 'react';
import { Button } from '@mantine/core';
import { faRedo, faTimes } from '@fortawesome/free-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'

const placeholder=`
  <span style="color: #828587; font-size: 10px; margin-left: 0.5em; margin-right: 0.5em">▼</span> 
  <span style="color: #828587;">Select a prompt or input your own</span> 
`
// Magic values - TODO: clean up in future
const autoText = "Write more tests similar to the tests saved below";
const selectExamplesText = "Write more tests similar to the selected tests saved below";

export default function PromptInput({value, onSubmit, disabled, dropdownOptions, style, isLoading}) {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [inputContent, setInputContent] = useState(placeholder);
  const [showError, setShowError] = useState(false);
  const [editable, setEditable] = useState(true);
  const contentRef = useRef(null);

  const optionMap = {};
  dropdownOptions.map(data => {
    const group = data.group ?? "";
    let groupList = optionMap[group];
    if (groupList != null) {
      groupList.push(data);
    } else {
      groupList = [data]
    }
    optionMap[group] = groupList;
  })

  function handleSubmit(e) {
    console.log("HANDLESUBMIT", contentRef, contentRef.current.innerText);
    const inputText = contentRef.current.innerText.trim();
    if (inputText === "" || inputContent === placeholder) {
      setShowError(true);
      return;
    } 
    
    let submitText = "";
    if (inputText === autoText) {
      submitText = "";
    } else if (inputText === selectExamplesText) {
      submitText = "Select examples";
    } else {
      submitText = inputText;
    }

    setShowError(false);
    onSubmit(e, submitText);
  }

  return (
    <>
      <div style={style}>
        <div style={{position: "relative"}} >
          { showError && <div style={{position: "absolute", top: "-20px", left: "0px", color: "red"}}>Please enter a value</div> }
          <div 
            contentEditable={editable}
            tabIndex={0}
            suppressContentEditableWarning={true}
            ref={contentRef}
            onClick={() => {
              if (inputContent === placeholder) {
                setInputContent("");
              }
            }}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === "Enter") {
                handleSubmit(e);
              }
            }}
            onChange={(e) => {
              console.log("PROMPTINPUT change event", e);
              setDropdownOpen(false);
            }}
            onFocus={() => setDropdownOpen(true)}
            onBlur={() => setDropdownOpen(false)}
            disabled={disabled}
            value={value}
            style={{
              width: "100%",
              padding: "0.5rem",
              textAlign: "left"
            }}
            dangerouslySetInnerHTML={{__html: inputContent}}
            />
            <button onClick={() => { setDropdownOpen(false); setInputContent("") }} style={{right: "5px", top: "10px", position: "absolute", border: "none", backgroundColor: "transparent", cursor: "pointer"}}>
              <FontAwesomeIcon icon={faTimes} style={{fontSize: "13px", color: "#333333", display: "inline-block"}} /> 
            </button>
        </div>
        { dropdownOpen ? 
          <div style={{
            position: "relative",
            marginTop: "0px" }}>
            <div style={{
              position: "absolute",
              background: "white",
              textAlign: "left",
              border: "1px solid #ccc",
              borderRadius: "5px",
              paddingLeft: "10px",
              paddingRight: "10px",
              paddingBottom: "15px",
              zIndex: "300",
              width: "100%" }}>
              {Object.keys(optionMap).map(group => (
                <div key={group}>
                  <div style={{
                    fontWeight: "bold",
                    borderBottom: "1px solid #333",
                    marginTop: "15px",
                    marginBottom: "10px" }}>
                    {group}
                  </div>
                  {optionMap[group].map((data, i) => (
                    <div key={group + "_option" + i}
                      className="adatest-hover-gray"
                      onMouseDown={(e) => {
                        // Avoid triggering the input's onBlur event
                        // https://stackoverflow.com/a/57630197
                        e.preventDefault();
                      }}
                      onClick={() => { 
                        console.log("User clicked on ", data);
                        setInputContent(data.view);
                        setEditable(data.editable ?? true);
                        setDropdownOpen(false);
                      }}
                      style={{
                        padding: "0.25rem 0.5rem",
                        cursor: "pointer"
                      }}>
                      { data.prefix && <span style={{ marginRight: "0.5rem" }}>{data.prefix}</span> }
                      <div style={{display: "inline-block"}} dangerouslySetInnerHTML={{__html: data.view}}/>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
          : null }
      </div>
      <Button style={{marginLeft: "10px", alignSelf: "end"}} onClick={handleSubmit}>
        <FontAwesomeIcon className={isLoading ? "rotating" : ""} icon={faRedo} style={{fontSize: "13px", color: "#FFFFFF", display: "inline-block"}} /> 
      </Button>
    </>
  )
}