# Book rating prediction - Emmanuel Gakosso - DSTI

## Link to the deployed app: https://emmanuel558-book-rating-predi-book-rating-prediction-app-2dmh9i.streamlit.app/

#### The deployment on streamlit is made available through Book_rating_prediction_app.py

### The aim of this project is to train a machine learning model to predict book rating and deploy it in a streamlit web app.
### All the steps from importing, data analysis, feature engineering, machine learning model and streamlit app will be explained within the notebook.

#### There are 3 main steps: 

#### Step 1: Reading dataset and core data analysis
#### Step 2: Feature engineering and data preprocessing (based on data analysis)
#### Step 3: Machine learning Model

#### I choose two models for this job: Linear Regression and Random Forest 

### The reasons: 
#### Linear Regression is a well-established and widely used method in predictive modeling for continuous data, which makes it a natural choice for predicting the average rating of a book.
#### It is a simple model that can easily be interpreted and explained, making it a good choice for understanding the factors that contribute to the rating of a book.
#### Additionally, the assumptions of linear regression are relatively easy to understand and diagnose, making it easy to evaluate the quality of the model and the reliability of its predictions.

#### However, in our dataset we have two variables highly correlated together : rating_counts and text_reviews_count. As we don't have many predictors, I choosed to keep both in my model. Multicollinearity can reduce the effectiveness of a linear regression model because it violates the assumptions of the linear regression model.  When this happens, it becomes difficult to distinguish the effect of each independent variable on the dependent variable, as their effects become confounded.

#### So I decided to give a try to Random Forest Regressor because unlike linear regression models, Random Forest Regressor is not affected by multicollinearity, as it does not assume a linear relationship between the independent and dependent variables.

#### I let your know to discover in the streamlit app what of both performs better. :wink:





