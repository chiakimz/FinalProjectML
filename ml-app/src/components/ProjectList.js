import React from 'react';
import PropTypes from 'prop-types';
function ProjectList (props) {
  return (
    <ul>
    {
      props.projects.map(project => (
        <li key={project.id}>
          {project.name}
        </li>
      ))
    }
    </ul>
  )
}
ProjectList.propTypes = {
  projects: PropTypes.array.isRequired
};
export default ProjectList;
