import React, { useState, useRef } from 'react';

export default function PromptInput({placeholder, value, onChange, disabled, dropdownOptions, style}) {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const inputRef = useRef();

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

  return (
    <div style={style}>
      <div style={{position: "relative"}}>
        <input 
          ref={inputRef}
          type="text"
          onChange={(e) => {
            setDropdownOpen(false);
            onChange(e.target.value);
          }}
          onFocus={() => setDropdownOpen(true)}
          onBlur={() => setDropdownOpen(false)}
          placeholder={placeholder}
          disabled={disabled}
          value={value}
          style={{
            width: "100%",
            padding: "0.5rem"
          }} />
      </div>
      { dropdownOpen ? 
        <div style={{
          position: "relative",
          marginTop: "10px" }}>
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
                {optionMap[group].map(data => (
                  <div key={data.value}
                    className="adatest-hover-gray"
                    onMouseDown={(e) => {
                      // Avoid triggering the input's onBlur event
                      // https://stackoverflow.com/a/57630197
                      e.preventDefault();
                    }}
                    onClick={() => { 
                      console.log("User clicked on ", data.value);
                      onChange(data.value);
                      setDropdownOpen(false);
                    }}
                    style={{
                      padding: "0.25rem 0.5rem",
                      cursor: "pointer"
                    }}>
                    { data.prefix && <span style={{ marginRight: "0.5rem" }}>{data.prefix}</span> }
                    <span>{data.value}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
        : null }
    </div>
  )
}