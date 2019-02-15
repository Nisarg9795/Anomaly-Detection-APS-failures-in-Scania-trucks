# Anomaly-Detection-APS-failures-in-Scania-trucks
We are trying to model a prediction system for detecting a component failure in the
Air Pressure System (APS) in the heavy Scania trucks. The prediction will tell us if there is an
imminent failure in the heavy trucks. The data were thus collected from the APS system that is
used in these day-to-day trucks. The APS, in general, is a system which generates pressurized air
to use in different component functions in the trucks such as braking, gears, suspension, etc. A
positive class is given to component failure that belongs to the APS system and a negative class
is given to component failure related to anything else. Other than just predicting the failure of the
component, we are also trying to optimize the cost of a failure. A cost of 10 is given to a correct
prediction i.e. predicting a failure of APS component and a cost of 500 is given to a false
negative i.e. to failure that was not predicted by our model. Thus, penalty minimizing is also one
of our main goals. The problem can be said to be a classification problem. We have used 5
different models, Logistic regression, Support Vector Machine, Random forest, Gaussian model,
Random model and K-Nearest Neighbour. All of these models use a different type of data
cleaning techniques and feature selection. In the end, according to the cost predicted by each
model, a weight is assigned to each one of them and a final decision is taken based on this
weighted polling. Our experiments show that the best classifier is the Random Forest with the
cost of around 10,000. Also, the accuracy of all the other models came out to be approximately
95%.
