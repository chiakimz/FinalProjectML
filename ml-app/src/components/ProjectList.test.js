import React from 'react';
import {shallow} from 'enzyme';
import ProjectList from './ProjectList';
import { configure } from 'enzyme';
import Adapter from 'enzyme-adapter-react-16';

// configure({ adapter: new Adapter() });
let mockProjects, wrapper;

beforeEach(() => {
  mockProjects = [
    {id: 1, name: 'Mock Project 1'},
    {id: 2, name: 'Mock Project 2'},
    {id: 3, name: 'Mock Project 3'},
  ];

  projectSelectFn = jest.fn();

  wrapper = shallow(
    <ProjectList
    projects={mockProjects}
    onProjectSelect={projectSelectFn}
    />
  );
});

afterEach(() => {
  productSelectFn.mockReset();
});
it('should display the project name in each "<li>" element', () => {
  const firstElement = wrapper.find('li').first();
  expect(firstElement.contains(mockProject[0].name)).toEqual(true);
});

it('should call "props.onProductSelect" when an <li> is clicked', () => {
  const firstElement = wrapper.find('li').first();
  expect(productSelectFn.mock.calls.length).toEqual(0);
  firstElement.simulate('click');
  expect(productSelectFn.mock.calls.length).toEqual(1);
  expect(productSelectFn.mock.calls[0][0]).toEqual(mockProducts[0]);
});
