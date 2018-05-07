import React, { Component } from 'react';
import ProjectList from './components/ProjectList';
class App extends Component {
constructor(props) {
    super(props);
    this.state = {
      projects: [
        {id: 1, name: 'Iris', brand: 'Supervised Learning'},
        {id: 2, name: 'Something', brand: 'Supervised Learning'},
        {id: 3, name: 'Something', brand: 'Supervised Learning'},
      ]
    }
  }
render() {
    return (
      <div>
        <h1>Wolves of Commercial Street Projects</h1>
        <ProjectList projects={this.state.projects}/>
      </div>
    );
  }
}
export default App;
