# Final Project
[Agnes Donat](https://github.com/agnesdonat) || [Chiaki Mizuta](https://github.com/chiakimz) || [George Drayson](https://github.com/GeorgeDrayson) || [Raefe Newton-Jones](https://github.com/Raefey)
> For the things we have to learn before we can do them, we learn by doing them.<br>
Aristotle

## Introduction

We are four software developers who have been teamed up based on our interest in Machine Learning. Our final project, delivered in just 9 days, is a series of bots trained with supervised learning that can predict events ranging from fraudulent transactions to types of Iris flowers to the likelihood of a tumour being malignant or benign. We are all passionate about Test Driven Development and well-crafted code, as well as following best practices of the SOLID principles taught at Makers Academy.

## Tech Stack

#### Backend

* Python
* TensorFlow

#### Frontend
 * TensorBoard

#### Testing

* unittest with TensorFlow's testing library

#### Libraries

* Matplotlib

## Getting started

1. Fork and clone this repository
2. Go into the iris folder
  ```
  cd iris
  ```
3. Start the virtual environment:
  ```
  source ./bin/activate
  ```
4. Download [TensorFlow](https://www.tensorflow.org/install/):
  ```
  pip3 install --upgrade tensorflow
  ```

## Running backend tests

1. cd to the folder of your choice then run the test file, e.g.:
```
python test_iris.py
```
## Training the bot

1. cd to the folder of your choice then run the training file, e.g.:
```
python iris_training_controller.py
```
## Process
### Week 1

  <strong>Monday:</strong> Started individual research on Machine Learning. We set up two Trello boards: one for sharing useful links to articles and videos and one for task delegation. Raefe also summarised ML concepts in a handy diagram: <br>

  ![Raefe's diagram for ML](./iris/public/ml_diagram.png "Raefe's diagram on ML")  

  <strong>Tuesday:</strong> In the morning, we reviewed each others' FizzBuzz code written in Python, tested with Pytest, and continued with more research. Later, we made a decision that instead of training a deep learning car with Reinforcement Learning, we would focus on Supervised learning.<br>

  <strong>Wednesday:</strong> Working in pairs, we read through TensorFlow's Eager Execution tutorial and used their example of the Iris flower dataset to categorising flower species. This gave us a better understanding of Tensorflow syntax and about the intricacies of supervised learning model<br>

  <strong>Thursday:</strong> Swapping pairs, we looked into testing the code we studied the previous day and finding a solution for serialising our Python object so we can save our trained bot. In order to tame the unstructured TensorFlow code, it had to be encapsulated into classess and fully tested. Using unittest with TensorFlow's testing library the production code became neatly organised with an Iris class and several methods each following the SRP. We were also experimenting with Python's pickle module, but eventually, we dropped this idea and used TensorFlow's Saver class. <br>

  <strong>Friday:</strong> George and Chiaki integrated persistent data into the project, while Raefe and Agnes added the ability to print graphs for the Loss and Accuracy.

 By the end of the week, we had a basic understanding of Machine Learning concepts, and a fully tested and trained Model for categorising Iris flowers that also returned its Loss and Accuracy results in graphs.

  ![Alt text](./iris/public/graphs.png)
