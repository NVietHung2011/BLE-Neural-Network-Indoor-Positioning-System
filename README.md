# BLE-Neural-Network-Indoor-Positioning-System
Indoor Positioning System using Bluetooth Low Energy Beacons, Raspberry Pi 3 board, based on RSS Fingerprinting and Neural Network algorithm.
The Neural Network (NN) in the system is a combination of 2 Neural Network modules, resposible for label classification in different grid sizes when dividing tracking area. The inputs are Received Signal Strength Indicators from 8 iBeacon devices installed around the experimental area. The system gives 2 outputs, the estimated area and the estimated position of the target.
The 1st NN has 2 hidden layers, 16 and 8 neurons, respectively. This module is for identify which area the target is in.
The 2nd NN also has 2 hidden layers, 16 and 40 neurons, respectively. This module takes in RSSs and the output of the 2st module as its inputs, and return the position of the target.
