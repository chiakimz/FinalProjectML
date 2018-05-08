import React from 'react';
import PropTypes from 'prop-types';
function ProjectList (props) {
  return (
    <ul>
      <li><a href="/iris">Iris</a></li>
      <li><a href="/fraud">Credit Card Fraud</a></li>
      <li><a href="/cancer">Cancer</a></li>
    </ul>
  )
}
ProjectList.propTypes = {
  projects: PropTypes.array.isRequired,
  onProjectSelect: PropTypes.func.isRequired
};
export default ProjectList;
