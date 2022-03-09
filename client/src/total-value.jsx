import React from 'react';
import autoBind from 'auto-bind';
import { get } from 'lodash';

export default class TotalValue extends React.Component {
    constructor(props) {
      super(props);
      autoBind(this);
  
      // our starting state 
      this.state = {
        // note that all the ids will also be properties of the state
      };
    }
  
    setSubtotal(id, subtotal) {
      let update = {};
      update[id] = subtotal;
      this.setState(update);
    }
  
    render() {
      // we just sum up the current active subtotals
      let total = 0;
      for (let i in this.props.activeIds) {
        total += get(this.state, this.props.activeIds[i], 0);
      }
      
      return <React.Fragment>
        {total}
      </React.Fragment>
    }
  }