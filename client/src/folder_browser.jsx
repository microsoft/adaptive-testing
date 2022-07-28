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

    moveFolder(e, key){
        if(this.props.not_move!=true){
            var cur_obs_state = JSON.stringify(this.state.observe_state)
            window.location.href = window.location.origin+key+'?obs='+cur_obs_state
        }else{
            var sel_idx = this.props.selected_concepts.indexOf(key)
            if(sel_idx==-1){
                this.props.selected_concepts.push(key)
                this.props.selected_concepts.sort()
                this.props.mother_this.setState({})
            
            }else{
                this.props.selected_concepts.splice(sel_idx, 1)
                this.props.mother_this.setState({})
            }
        }
        // this.props.history.push(this.props.prefix + key)
        // this.props.onOpen(key);
        
    }

    DragOver(e){
        e.preventDefault()
    }

    Drop(e, key){
        const id = e.dataTransfer.getData("id");
        console.log('drop', id, key, this.props.mother_this.state.hovered_part)
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

    mouseEnter(e, key){
        this.props.mother_this.setState({hovered_concept: key, hovered_part: 'folder'})
    }

    mouseOut(e, key){
        this.props.mother_this.setState({hovered_concept: undefined, hovered_part: undefined, dropped:false})
    }


    // dragOut(e, key){
    //     this.props.mother_this.setState({hovered_part: undefined})
    // }



    renderFolder(structure){
        console.log('structure:', structure)
        return Object.keys(structure).map((key, idx)=>{
            var k_list = key.split('/')

            var bc = ''
            if(this.props.not_move!=true){

            }else{
                if(this.props.hovered_concept==key){
                    bc = '#cfebff'
                }else if(this.props.selected_concepts.indexOf(key)!=-1){
                    bc = '#9fd7ff'
                }
            }
            if (key=='count'){} // edit here charvi
            else{
            return (
            <div> 
                <div onDrop={e => this.Drop(e, key)} onDragOver={e => this.DragOver(e)}
                onMouseEnter={e => this.mouseEnter(e, key)} onMouseOut={e => this.mouseOut(e, key)}
                    style={{backgroundColor:bc, borderRadius: 5, padding: 2}}> 

                    {Object.keys(structure[key]).length>1 && <span onClick={e => this.toggleItem(e, key)} onMouseEnter={e => this.mouseEnter(e, key)} onMouseOut={e => this.mouseOut(e, key)}>{this.state.observe_state[key]?"▾":"▸"}</span>}
                    {Object.keys(structure[key]).length==1 && <span style={{opacity:0}}>{this.state.observe_state[key]?"▾":"▸"}</span>}
                    <span onClick={e => this.moveFolder(e, key)} onDrop={e => console.log('drop', key)} onMouseEnter={e => this.mouseEnter(e, key)} onMouseOut={e => this.mouseOut(e, key)}>
                    <FontAwesomeIcon icon={faFolder} style={{fontSize: "14px", marginRight:'3px', color: "rgb(84, 174, 255)", display: "inline-block"}} />    
                        {k_list[k_list.length-1]}
                    </span> 
                     <span style={{color:"green"}}> {structure[key]['count'][0]} </span> <span> / </span>
                        <span style={{color:"red"}}> {structure[key]['count'][1]} </span>
                </div>
                <div style={{marginLeft: '10px'}}>
                    {structure[key]!=undefined && Object.keys(structure[key]).length>0 && this.state.observe_state[key] && 
                    this.renderFolder(structure[key])}
                </div>
            </div>)
        }
        })
    }

    render(){
        return (<div style={{textAlign:'left', padding: '5px'}} className={"unselectable"}>
            <div onClick={e => this.moveFolder(e, '')}>Home</div>
            {this.props.structure!=undefined && this.renderFolder(this.props.structure)}
        </div>)
    }
}