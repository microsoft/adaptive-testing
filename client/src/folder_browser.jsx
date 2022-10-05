import React from 'react'
import autoBind from 'auto-bind';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {faFolder, faFolderPlus} from '@fortawesome/free-solid-svg-icons'
import QuestionMarkFolder from "./QuestionMarkFolder"
import { Button } from '@mantine/core';

export default class FolderBrowser extends React.Component{
    constructor(props){
        super(props);
        autoBind(this);

        this.state = {
            observe_state: {}
        }
    }

    componentDidMount(){
        console.log(this.gup('obs'))
        if(this.gup('obs')!=null){
            this.state.observe_state = JSON.parse(decodeURI(this.gup('obs')))
        }else{
            this.structureObs(this.props.structure)
        }
        this.setState({})
        // for(var k in this.props.structure){
        //     console.log(this.gup('obs'))
        //     if(this.gup('obs')!=undefined){
        //         observe_state = JSON.parse(this.gup('obs'))
        //     }else{

        //         observe_state[k]=false    
        //     }
            
        // }
    }

    structureObs(structure){
        for(var k in structure){
            this.state.observe_state[k] = false
            this.structureObs(this.props.structure[k])
        }
        
        
    }

    gup( name, url ) {
        if (!url) url = location.href;
        name = name.replace(/[\[]/,"\\\[").replace(/[\]]/,"\\\]");
        var regexS = "[\\?&]"+name+"=([^&#]*)";
        var regex = new RegExp( regexS );
        var results = regex.exec( url );
        return results == null ? null : results[1];
    }

    toggleItem(e, key){
        console.log(key, this.state.observe_state[key])
        this.state.observe_state[key] = !this.state.observe_state[key]
        this.setState({})
    }

    changeTopic(e, key){
        this.props.onClick(key);
    }

    DragOver(e){
        e.preventDefault()
    }

    Drop(e, key){
        const id = e.dataTransfer.getData("id");
        let suffix = "";
        if (id.includes("/")) {
            suffix = "/" + id.split("/").pop();
        }
        this.props.onDrop(id, {topic: key + suffix});
        // console.log('drop', id, key, this.props.mother_this.state.hovered_part)
        // var key_split = key.split('/')
        // var cur_key = ''
        // var k = ''
        // var concept_origin = '/'+key_split[1]
        // for(var i in key_split){
        //     if(i==0){continue}
        //     cur_key = cur_key + '/'+key_split[i]
        //     k = k + cur_key
        //     if(i!=key_split.length-1){k=k+'|'}
        // }
        // this.props.onDrop(id, {topic:k, type:'data_in_out|'+concept_origin})
    }

    // mouseEnter(e, key){
    //     this.props.mother_this.setState({hovered_concept: key, hovered_part: 'folder'})
    // }

    // mouseOut(e, key){
    //     this.props.mother_this.setState({hovered_concept: undefined, hovered_part: undefined, dropped:false})
    // }


    // dragOut(e, key){
    //     this.props.mother_this.setState({hovered_part: undefined})
    // }



    renderFolder(structure){
        console.log('structure:', structure);
        const currentTopic = this.props.currentTopic;
        return Object.keys(structure).map((key, idx)=>{
            // TODO: Restructure the structure object to have a count field & a folders array
            if (key === "count") {
                return null
            }
            const k_list = key.split('/');
            const isSelected = currentTopic === key;
            const backgroundColor = isSelected ? '0078D4' : null;
            const textColor = isSelected ? 'FFFFFF' : '000000';
            const folderRowClass = isSelected ? null : 'adatest-folder-hover';
            const passColor = isSelected ? '#7bdd7b' : '#09b909';
            const failColor = isSelected ? '#ef6d6d' : '#eb0000';
            const numFolders = structure[key] != null ? Object.keys(structure[key]).length : 0
            const testCounts = structure[key]['count']
            const passCount = testCounts != null ? testCounts[0] : 0
            const failCount = testCounts != null ? testCounts[1] : 0
            const isNotSureFolder = key === "/Not Sure";
            const iconStyle = {
                fontSize: "14px",
                marginRight:'3px',
                color: "rgb(84, 174, 255)",
                display: "inline-block"
            }

            return (
                <div>
                    <div onDrop={e => this.Drop(e, key)} onDragOver={e => this.DragOver(e)}
                        className={folderRowClass}
                        style={{backgroundColor: backgroundColor, padding: 2}}> 

                        { numFolders > 1 && <span style={{cursor: "pointer"}} onClick={e => this.toggleItem(e, key)}>{this.state.observe_state[key]?"▾":"▸"}</span>}
                        <span style={{cursor: "pointer", marginLeft: numFolders > 1 ? "0px" : "11px", color: textColor}} onClick={e => this.changeTopic(e, key)} onDrop={e => console.log('drop', key)}>
                            { isNotSureFolder ? 
                                <QuestionMarkFolder style={iconStyle} />
                                :
                                <FontAwesomeIcon icon={faFolder} style={iconStyle} /> }
                            {k_list[k_list.length-1]}
                        </span> 
                        { !isNotSureFolder && 
                        <>
                            <span style={{color: passColor}}> {passCount} </span>
                            <span> / </span>
                            <span style={{color: failColor}}> {failCount} </span>
                        </>
                        }
                    </div>
                    <div style={{marginLeft: '10px'}}>
                        {numFolders > 0 && this.state.observe_state[key] && this.renderFolder(structure[key])}
                    </div>
                </div>
            )
        })
    }

    render(){
        const iconStyle = {
            fontSize: "14px",
            marginRight:'3px',
            color: "rgb(84, 174, 255)",
            display: "inline-block"
        }
        return (
            <div style={{textAlign:'left'}} className={"unselectable"}>
                <div className='adatest-title' style={{cursor: "pointer", display: "flex", flexDirection: "row"}} 
                    onClick={e => this.changeTopic(e, '')} 
                    onDrop={e => this.Drop(e, '')}
                    onDragOver={e => this.DragOver(e)}>
                    <span style={{alignSelf: "start", paddingTop: "10px", marginRight: "20px"}}>Topics</span>
                    <Button color="gray" style={{alignSelf: "end"}} onClick={this.props.handleClick}>
                         <FontAwesomeIcon icon={faFolderPlus} style={{fontSize: "13px", color: "#FFFFFF", display: "inline-block"}} />
                    </Button>
                </div>
                
                {this.props.structure!=undefined && this.renderFolder(this.props.structure)}
            </div>
        )
    }
}