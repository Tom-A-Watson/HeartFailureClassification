# HeartFailureClassification

In the second year of university, I had an assignment for the module 'Intelligent Systems 1' called '**Open Ended Project**'. This project involved finding a dataset for a classification/ regression problem, training & testing pre-existing models on the dataset, evaluating performance, and fine-tuning the system to (potentially) increase the models' performances. In case, I chose classification as there is a lot more to talk about. Originally, I went with a 'Mushroom classification' dataset, however this was working *too* well for this assignment, because each classifier was attaining perfect scores, meaning no comparisons or analysis could be made. Thus, after searching further for applicable datasets, I came across the 'Heart Failure' dataset. I obtained these datasets from an amazing machine-learning repository called [Kaggle](https://www.kaggle.com/), and would highly recommend visiting the site if you're interested in machine-learning. Lastly, I was required to write a report about how I went about obtaining the dataset, trained and tested the various models etc. If you wish, feel free to read-through my analysis and give feedback 😄

## Open Ended Project

### Evidence of Understanding
#### What is the functionality of your system? 
The sole functionality of my system of algorithms/models is to accurately predict whether a person has heart disease or not. This is based on various types of data collected from hospitals. 

#### What are the basic inputs and outputs?
My chosen dataset is comprised of 5 separate heart datasets from various locations across the world, making it the largest publicly-available dataset for researching heart disease. It contains 12 types/columns of clinical data from 918 different patients [2], including ‘HeartDisease’ which is this dataset’s target. I.e., the deciding factor/column of whether a patient has heart disease or not.
	
**Clinical data types**:

+ *Age*
+ *Sex* 
+ *Serum cholesterol*
+ *Chest pain type* 
+ *Maximum heartrate achieved* 
+ *Resting blood pressure*	
+ *Fasting blood sugar*
+ *Resting electrocardiogram results*
+ *Exercise-induced angina*
+ *Oldpeak (numerical ST equivalent)*
+ *Slope of the peak exercise ST segment*
+ *HeartDisease (**target**)*

These 12 columns are the basic inputs of the system, since each model utilizes them in order to form conclusions. This is done by splitting the data randomly into two separate datasets, one for training the models and the other for testing the models. This will be talked about further in the Analysis and preprocessing section in the algorithm explanation. Once the predictions are evaluated, the result (output of the system) is a printed statement in a terminal, giving an accuracy and precision score ranging from 0 to 1 for each algorithm. 

#### What real-world problem does it solve?
Ideally, the objective is to get the percentage of correct predictions to be as high as possible, as lives could potentially be at risk from an incorrect one. This is due to the real-world usage, which (if the accuracy was almost 100%) would be to implement the system in hospitals across the globe to aid in early detections of heart disease. Implementing this would ultimately solve the problem of doctors needing to analyze extrapolated data from hundreds of tests for many hours in order to come to a conclusion. It solves this problem because the system of algorithms can analyze the data and come to conclusions of similar accuracy in a matter of seconds/minutes. In other words, machine learning techniques are useful not-only for faster detections of heart disease, but the automation of this itself removes the need for hours or even days of manual analysis from healthcare professionals.


