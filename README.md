# Self Driving Car AI
This is an attempt to make a simple Self Driving Car AI. The AI uses [Deep Q-Learning](https://en.wikipedia.org/wiki/Q-learning), a Reinforcement Learning technique, which tells an agent which action to take under which circumstances. It does not require a model of the environment and can handle problems with stochastic transitions and rewards, without requiring adaptations. AI has been simulated and tested using GUI application created with Kivy.

## Dependencies
1. Pytorch
2. kivy
3. Numpy

## Steps
Run ` python map.py `

## Simulation
The car's source and destination alternates between top-left and bottom-right corner. Once it reaches bottom-right corner it's destination automatically changes to top-left corner and vice versa. The yellow lines and blobs are bascially the borders and hurdles in the car's path which it has to avoid in order to maximise the rewards. The borders and hurdles are stochastic transitions, this shows that the car does not need familiar environments.

The 3 colored circles around the car are the sensors that sense the whether there is any yellow content in the proximity of the car. The more the yellow content more intense will be the sensed signal.

![Alt Text](https://github.com/Minkush-sopho/SelfDrivingCarAI/blob/master/ScreenCapture_26-07-2018%2012.15.23_23.gif)
