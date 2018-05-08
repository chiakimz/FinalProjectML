import React, { Component } from 'react';
import ProjectList from './components/ProjectList';

class App extends Component {

constructor(props) {
    super(props);
    this.state = {
      selectedProjects: [],
      projects: [
        {id: 1, name: 'Iris', brand: 'Supervised Learning'},
        {id: 2, name: 'Fraud', brand: 'Supervised Learning'},
        {id: 3, name: 'Diabetes', brand: 'Supervised Learning'},
      ]
    }
  }

handleProjectSelect (project) {
  this.setState(prevState => {
    return {
      selectedProjects: prevState.selectedProjects.concat(project)
    }
  });
}

render() {
    return (
      <div>
        <h1>Wolves of Commercial Street Projects</h1>
        <p>You have selected {this.state.selectedProjects.length}
project(s).</p>
        <ProjectList
         projects={this.state.projects}
         onProjectSelect={this.handleProjectSelect.bind(this)}
         />
      </div>
    );
  }
}
export default App;
