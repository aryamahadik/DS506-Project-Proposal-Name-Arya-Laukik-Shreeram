Project Proposal: Predicting Bank Customer Churn
Names: Arya Mahadik, Laukik Khade, Sreeram Maguluri DS506

Introduction

In this project we are trying to find why customers leave a bank. Basically We want to look at customer data and figure out who is likely to close their account known as churning and who is likely to stay. By analyzing things like credit scores, age and balance we can find patterns when a customer might be unhappy or to leave

Goals:
We want to build a model that can predict whether a specific bank customer will churn or not, based on the profile and account history this will help the bank identify at risk customers early so they can try to keep them.

Dataset:

We will use a dataset that is already available which contains information of the bank customers .The data looks like this:

	•	customer id
	•	credit score
	•	country
	•	gender
	•	age
	•	tenure
	•	balance
	•	Number of products
	•	Credit card status
	•	Active member status
	•	estimated salary
	•	churn 1 for yes, 0 for no



Project Timeline:

Week 1 - Download the dataset and clean the data (handle missing values, encode categorical variables like country and gender).
Week 2 - Visualize the data using bar charts, histograms, and heatmaps.
Week 3 - Build and train the models (Logistic Regression, Random Forest, XGBoost).
Week 4 - Test the models and compare results using accuracy, precision, and recall.

Data Collection Methods:

We will download the dataset directly from Kaggle. Before modeling, we will clean the data by checking for missing values and converting text columns like country and gender into numbers so the models can understand them.

Modeling:

To solve this we plan to test a few different classification models to predict a yes/no outcome (churn or no churn)

	•	Logistic Regression a good starting point to see simple relationships.
	•	Decision trees or Random forest are great for handling mix of data types and usually give better accuracy
	•	Xgboost If we need more power this is often the best for this kind of table data.

Visualization:

Before the modeling we want to visualize the data to understand it better.For example;
	•	Bar Charts to compare churn rates between different countries or genders.
	•	Histograms to see the distribution of age and credit scores for people who left with comparison to the people who stayed
	•	Correlation Heatmap to see if things such as balance and salary are related or not.

Test plan:	

	•	Training Set (80%)  will be used to teach the model patterns.
	•	Testing Set (20%) will be used to test the model on data which it hasn't seen before to check its accuracy.
	•	
We will look at evaluation metrics like accuracy precision and recall to decide if the model is good enough to use

Reference Link:

Dataset:

https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers


