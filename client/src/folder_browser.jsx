import React from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {faFolder} from '@fortawesome/free-solid-svg-icons'

export default class FolderBrowser extends React.Component{
    constructor(props){
        super(props);

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
        // console.log('drop', id, key, this.props.mother_this.state.hovered_part)
        var key_split = key.split('/')
        var cur_key = ''
        var k = ''
        var concept_origin = '/'+key_split[1]
        for(var i in key_split){
            if(i==0){continue}
            cur_key = cur_key + '/'+key_split[i]
            k = k + cur_key
            if(i!=key_split.length-1){k=k+'|'}
        }
        this.props.onDrop(id, {topic:k, type:'data_in_out|'+concept_origin})
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
        console.log('structure:', structure)
        return Object.keys(structure).map((key, idx)=>{
            // TODO: Restructure the structure object to have a count field & a folders array
            if (key === "count") {
                return null
            }
            const k_list = key.split('/')
            const bc = ''

            // TODO: Color background based on selected folder (current topic)
            // if(this.props.hovered_concept==key){
            //     bc = '#cfebff'
            // } else if (this.props.selected_concepts.indexOf(key)!=-1) {
            //     bc = '#9fd7ff'
            // }

            const numFolders = structure[key] != null ? Object.keys(structure[key]).length : 0
            const testCounts = structure[key]['count']
            const passCount = testCounts != null ? testCounts[0] : 0
            const failCount = testCounts != null ? testCounts[1] : 0

            return (
                <div>
                    <div onDrop={e => this.Drop(e, key)} onDragOver={e => this.DragOver(e)}
                        style={{backgroundColor:bc, borderRadius: 5, padding: 2}}> 

                        { numFolders > 1 && <span style={{cursor: "pointer"}} onClick={e => this.toggleItem(e, key)}>{this.state.observe_state[key]?"▾":"▸"}</span>}
                        <span style={{cursor: "pointer", marginLeft: numFolders > 1 ? "0px" : "11px"}} onClick={e => this.changeTopic(e, key)} onDrop={e => console.log('drop', key)}>
                            <FontAwesomeIcon icon={faFolder} style={{fontSize: "14px", marginRight:'3px', color: "rgb(84, 174, 255)", display: "inline-block"}} />
                            {k_list[k_list.length-1]}
                        </span> 
                        <span style={{color:"green"}}> {passCount} </span>
                        <span> / </span>
                        <span style={{color:"red"}}> {failCount} </span>
                    </div>
                    <div style={{marginLeft: '10px'}}>
                        {numFolders > 0 && this.state.observe_state[key] && this.renderFolder(structure[key])}
                    </div>
                </div>
            )
        })
    }

    render(){
        return (
            <div style={{textAlign:'left', padding: '5px'}} className={"unselectable"}>
                <div style={{cursor: "pointer"}} onClick={e => this.changeTopic(e, '')}>Home</div>
                {this.props.structure!=undefined && this.renderFolder(this.props.structure)}
            </div>
        )
    }
}