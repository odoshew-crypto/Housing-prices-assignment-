Housing Prices Prediction
Project 
This project focuses on predicting housing prices using machine learning regression models. The dataset helps analyze how different features such as income, house age, and location affect house prices

 Dataset
The dataset used in this project is the **California Housing Dataset**, which is available in the `scikit-learn` library.

You can load it using:

python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
df = housing.frame

Correlation Heatmap
A heatmap was used to understand relationships between features.
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

Income vs House Prices
This shows how median income affects house prices.
sns.scatterplot(x=df['MedInc'], y=df['MedHouseVal'])

House Price Distribution
This shows how house prices are distributed across the dataset.
sns.histplot(df['MedHouseVal'], bins=30)

Regression Plot
This shows the trend between income and house prices with a regression line.
sns.regplot(x=df['MedInc'], y=df['MedHouseVal'])
Machine Learning Models Used

Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
To build a regression model that predicts house prices based on features like income, location, and population.
Evaluation Metrics also i used
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R² Score
The project demonstrates how different features influence housing prices and how machine learning models can be used to make accurate prediction
