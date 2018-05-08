import React from 'react';
import {shallow} from 'enzyme';
import ProjectList from './ProjectList';
import { configure } from 'enzyme';
import Adapter from 'enzyme-adapter-react-16';

configure({ adapter: new Adapter() });

it('should render a list of products as an unordered list', () => {
  const mockProjects = [
    {id: 1, name: 'Iris', brand: 'Supervised Learning'},
    {id: 2, name: 'Fraud', brand: 'Supervised Learning'},
    {id: 3, name: 'Diabetes', brand: 'Supervised Learning'},
  ];
  const wrapper = shallow(<ProjectList projects={mockProjects}/>);
  expect(wrapper.find('li').length).toEqual(mockProjects.length); // 3
});
