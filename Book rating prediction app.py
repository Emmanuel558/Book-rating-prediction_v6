import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import pearsonr

from category_encoders import TargetEncoder
from sklearn.preprocessing import RobustScaler
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title("Exploratory Data Analysis and Detailed analysis of machine learning results")

dataset = st.container()
dataviz = st.container()
feature_engineering = st.container()
machine_learning_analysis = st.container()


with dataset:

    data = pd.read_csv('books.csv', error_bad_lines=False)
    # error_bad_lines argument allows to handle some lines for example these where commas are contained in authors names. This causes a column shift for the rows where this is the case.
    # For example: line with bookID 12224.

    st.header("First look at the data we have.")
    st.write(data.head(10))

    # Checking null values or duplicated values in dataset

    st.write(data.info()) # null values

    # Little cleansing on the columns fields
    data.columns = data.columns.str.replace(' ', '')

    # We don't have any missing values in our fields

    # duplicated values
    st.write('------------------------')
    st.write(
        f"We have {data.duplicated().sum()} number(s) of duplicated row(s) in the dataset.")
    st.write("After reading data description and metadata, we note that fields bookID, isbn and isbn13 contain same information.")
    st.write("The aim of these fields is to give a primary key for each row. So we decide to delete them in order to gain speed and to focus on columns that can be used to enrich the model or that have strong dependencies with the ratings column.")

    data = data.drop(['bookID', 'isbn', 'isbn13'], axis=1)

    with dataviz:

        #########################################

        st.subheader(
            "We explore the total ratings_count by title to take knowledge about the most reviewed books.")

        best_ratings_count_books = data.nlargest(5, 'ratings_count')

        fig = go.Figure(data=go.Bar(
            x=best_ratings_count_books['title'], y=best_ratings_count_books['ratings_count']))
        fig.update_layout(title='Top 5 Books with Highest Ratings Count',
                          xaxis_title='Book Title',
                          yaxis_title='Ratings Count',
                          font=dict(size=16),
                          width=1000, height=800)
        st.plotly_chart(fig)

        #########################################

        st.subheader(
            "We explore the total text reviews count by title to take knowledge about books with most text reviews count")

        best_text_reviews_count = data.nlargest(5, 'text_reviews_count')

        fig2 = go.Figure(data=go.Bar(
            x=best_text_reviews_count['title'], y=best_text_reviews_count['text_reviews_count']))
        fig2.update_layout(title='Top 5 Books with Highest Number of text reviews',
                           xaxis_title='Book Title',
                           yaxis_title='Ratings Count',
                           font=dict(size=16),
                           width=1000, height=800)
        st.plotly_chart(fig2)

        st.write(
            "Conclusion: Books with many text_reviews have also lots of rating count")

        #########################################

        st.subheader(
            "Now, let's explore the correlation between these two variables and average rating.")

        fig3 = px.scatter(data, x='average_rating',
                          y='ratings_count', width=900, height=600)

        fig4 = px.scatter(data, x='average_rating',
                          y='text_reviews_count', width=900, height=600)

        st.plotly_chart(fig3)
        st.plotly_chart(fig4)

        st.write("Conclusion of above analysis: The main part of books are located under 20000 text_reviews and rating_counts apart from some others as the five we saw aove")

        ###########################################

        st.subheader("Let's now analyse the distribution of number of pages.")

        fig5 = go.Figure()
        fig5.add_trace(go.Box(y=data['num_pages'], name='Number of pages'))
        fig5.update_layout(title='Distribution of number of pages in a given book.',
                           yaxis_title='num_pages',
                           font=dict(size=15),
                           width=1000, height=800)
        st.plotly_chart(fig5)

        st.write(
            "The number of pages field contains a lot of outliers after about 800 pages.")
        ##########################################

        st.subheader(
            "Now we get a look at the relationship between number of pages and average rating (our target variable).")

        fig6 = px.scatter(data, x='average_rating', y='num_pages')
        fig6.update_layout(title='Relation between average_rating and number of pages',
                           yaxis_title='num_pages',
                           font=dict(size=15),
                           width=1000, height=800)

        st.plotly_chart(fig6)

        st.write("After analyzing the distribution of num_pages and the relationship between it and average_rating, we can say that the books with num_pages above 1000 are outliers and can can negatively affect the quality of the model seeing the fact that most of books contain under 1000 pages.\n"
                 "Same conclusion for rating_counts above 1e6 and text_reviews_count above 20000")

        ############################################

        st.subheader("Let's analyse the distribution of target variable")

        fig7 = go.Figure(
            data=[go.Histogram(x=data['average_rating'], nbinsx=20)])
        fig7.update_layout(title='Distribution of Average Ratings',
                           xaxis_title='Rating',
                           yaxis_title='Count',
                           font=dict(size=16),
                           width=900, height=600)
        st.plotly_chart(fig7)

        st.write("The conclusion of analysis: People rate books on a scale of mainly 3 to 4. Very few if any books are rated at 0, 1 and 2 and a very small minority also rate at 5.")

        ###########################################

        st.subheader("Correlation analysis using heatmap.")

        correlation_matrix = data.corr().round(2)

        fig8 = px.imshow(correlation_matrix, color_continuous_scale='RdBu_r',
                         labels=dict(x='Variables', y='Variables',
                                     color='Corrélation'),
                         width=900, height=600)

        fig8.update_layout(title='Matrice de corrélation des variables numériques',
                           font=dict(size=16))

        st.plotly_chart(fig8)

        st.write("All explanatory variables are weakly correlated to the target variable. Text reviews count and ratings count are highly correlated.")

        #########################################

        st.subheader('Pearson Correlation Tests')

        # Pearson correlation test between average_rating and num_pages
        st.write('Correlation between average_rating and num_pages')
        pearson_coeff, p_value = pearsonr(
            data['num_pages'], data['average_rating'])
        st.write(pd.DataFrame(
            {'pearson_coeff': [pearson_coeff], 'p-value': [p_value]}))

        # Pearson correlation test between text_reviews_count and average_rating
        st.write('Correlation between text_reviews_count and average_rating')
        pearson_coeff, p_value = pearsonr(
            data['text_reviews_count'], data['average_rating'])
        st.write(pd.DataFrame(
            {'pearson_coeff': [pearson_coeff], 'p-value': [p_value]}))

        # Pearson correlation test between ratings_count and average_rating
        st.write('Correlation between ratings_count and average_rating')
        pearson_coeff, p_value = pearsonr(
            data['ratings_count'], data['average_rating'])
        st.write(pd.DataFrame(
            {'pearson_coeff': [pearson_coeff], 'p-value': [p_value]}))

        st.write("Assuming that our null hypothesis is that two selected variables are independant in dataset.\n"
                 "There is less than a 5% chance that the observed correlation between the explanatory variables and book evaluation is due to chance. In other words, this suggests that there is a strong correlation between the two variables studied. The correlation is statistically significant.\n"
                 "Although the numerical value of pearson correlation is low.")

        #######################################

        st.subheader(
            "Analysis of categorical variables publisher, language_code and authors")

        # Publisher

        st.write('List of some of the best publishers')
        st.write(data.groupby(['publisher'])[
                 'average_rating'].mean().sort_values(ascending=False).head(20))

        # How many distinct publishers we have ?

        st.write('-------------------------------------')
        st.write(
            f"There are {data['publisher'].nunique()} unique publishers in our dataset.")
        st.write("According to the large number of publishers, it's not suitable to build a chart to see the distribution of each of them.\n"
                 "But for the analysis of output of machine learning model, we keep in mind the name of some of the best of them.")

        # Language code

        language_counts = data['language_code'].value_counts()
        fig9 = px.pie(language_counts, values=language_counts.values, names=language_counts.index,
                      title='Language Code Distribution', width=900, height=600)

        fig9.update_traces(textposition='inside', textfont_size=20)
        st.plotly_chart(fig9)

        st.write("The majority of books are written in english.")

        # Authors

        st.write("In order to recommand a special author for a given book, let's analyze the authors with many ratings_count")

        authors = data.nlargest(5, ['ratings_count'])
        fig10 = px.bar(authors, x='ratings_count', y='authors', orientation='h',
                       title='Authors with highest ratings_count')

        st.plotly_chart(fig10)

    with feature_engineering:

        # Categorical variables

        st.subheader(
            "Feature engineering and data preprocessing (based on data analysis)  -   Categorical variables")

        st.write("Categorical variables")

        st.write("Publication_date : The date of publication of a book may be important for understanding the context in which it was written and for appreciating its historical or cultural significance, but it does not necessarily determine whether the book is appreciated or not.\n"
                 "So we decide to delete this field.\n")
        st.write("Publisher: The publisher of a book is not a reliable indicator of whether a book is popular or not. Although some publishers have a reputation for publishing high quality books, there are many factors that influence the popularity of a book, such as the author, subject matter, writing style, promotion, etc.\n "
                 "In addition, a book may be well received by the public even if it is published by a small, little-known publisher, whereas a book published by a large publisher may not be as successful. So let's delete this field too.\n")
        st.write("title: We have more than 10.000 occurrences of different title in dataset.\n"
                 "One-hot encoding is not suitable because the number of variables will be huge and labelencoding is not suitable because we do not have an inherent order.\n")

        st.write("In view of what has been said, we decide to delete from dataset variables publication_date, publisher and title.")

        data = data.drop(['publication_date', 'publisher', 'title'], axis=1)

        st.subheader("Mean-based encoding on categorical variables")

        st.write("Explanation of our variable encoding method:\n"
                 "Target encoding, also known as likelihood encoding, is a method for encoding categorical variables in machine learning. It replaces the original categorical variable with the average of the target variable for each category.\n"
                 "One advantage of target encoding is that it can capture information about the relationship between the categorical variable and the target variable, which can be useful for certain models. It can also help to reduce the dimensionality of the dataset compared to one-hot encoding, \n where each category is represented by a separate binary feature. This can be beneficial when dealing with a large number of categories"
                 "As we have many categories in our string fields, one-hot encoding will result in too more variables and labelEncoder can easily introduce noise in the model.\n"
                 "This is because it can introduce an arbitrary order of the categories which may not reflect any meaningful relationship between them.\n"
                 "This can be problematic for certain models, such as linear regression, where the order of the categories can affect the estimated coefficients and lead to incorrect predictions.")
        

        te_authors = TargetEncoder()

        data['authors'] = te_authors.fit_transform(
            data['authors'].values, data['average_rating'].values)

        te_language = TargetEncoder()

        data['language_code'] = te_language.fit_transform(
            data['language_code'].values, data['average_rating'].values)

        st.write("Look at the data encoded for authors and language code encoding.")

        styled_table = styled_table = data.style.format({'font-size': '14px'})


        st.write(styled_table)

        # Numerical variables

        st.subheader(
            "Feature engineering and data preprocessing (based on data analysis)  -   numerical variables")

        st.write("Remove outliers: As we saw during data analysis, there are few outliers in fields rating_counts, text_reviews_count and num_pages.\n"
                 "So we decided to fix some treshold to delete rows with outliers.")

        treshold_rating_counts = 800000
        treshold_text_reviews_count = 20000
        treshold_num_pages = 1200

        data = data[data['ratings_count'] <= treshold_rating_counts]
        data = data[data['text_reviews_count'] <= treshold_text_reviews_count]
        data = data[data['num_pages'] <= treshold_num_pages]

        # Normalisation of numerical variables to remove the effect of scaling values that are different between these columns

        numerical_features = ['num_pages',
                              'ratings_count', 'text_reviews_count']

        # create a RobustScaler object
        scaler = RobustScaler()

        # fit the scaler to the numerical features
        scaler.fit(data[numerical_features])

        # transform the numerical features using the scaler
        scaled_features = scaler.transform(data[numerical_features])

        # replace the original features in the dataframe with the scaled features
        data[numerical_features] = scaled_features

        st.write("Look at the data normalized using RobustScaler to use median which is less sensitive than average to outliers and after removing outliers.")

        styled_table_2 = data.style.format({'font-size': '14px'})

        st.write(styled_table_2)

    with machine_learning_analysis:

        X = data[['num_pages', 'ratings_count',
                  'text_reviews_count', 'authors', 'language_code']]
        y = data['average_rating']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=23)

        st.subheader("Linear Regression Model")

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        predictions = lr.predict(X_test)

        st.subheader("Analysis of the results of trained model")

        pred = pd.DataFrame({'Actual': y_test.tolist(),
                            'Predicted': predictions.tolist()}).head(15)

        styled_table_3 = pred.style.format({'font-size': '14px'})

        st.write(styled_table_3)

        # Checking results

        st.subheader("Checking of linearity")

        fig11 = px.scatter(x=y_test, y=predictions,
                           title='Actual Vs Predicted value - Linear Regression')
        fig11.update_layout(
            xaxis_title='Actual value',
            yaxis_title='Predicted value'
        )

        st.plotly_chart(fig11)

        st.subheader(
            "Evaluation of model performance by using linear regression.")

        st.write('MAE - Linear Regression:',
                 metrics.mean_absolute_error(y_test, predictions))
        st.write('MSE - Linear Regression:',
                 metrics.mean_squared_error(y_test, predictions))
        st.write('RMSE - Linear Regression:',
                 np.sqrt(metrics.mean_squared_error(y_test, predictions)))
        st.write('R2 - Linear Regression:', lr.score(X, y))

        st.write("--------------------------------------------")
        st.write("Conclusion of metrics evaluation")
        st.write(f"The R2 is about {round(lr.score(X,y),2)} so {round(lr.score(X,y)*100,2)}% of the variability in the dependent variable can be explained by the independent variables included in the model.")
        st.write(f"The MAE is about {round(metrics.mean_absolute_error(y_test, predictions),2)} so on average, the model's predictions are off by about {round(metrics.mean_absolute_error(y_test, predictions),2)} units of the target variable.")

        st.subheader("Random Forest regressor Model")

        rf_reg = RandomForestRegressor(n_estimators=250, random_state=44)
        rf_reg.fit(X_train, y_train)

        # Predicting on Test Data
        y_pred = rf_reg.predict(X_test)

        # Checking of linearity

        fig12 = px.scatter(
            x=y_test, y=y_pred, title='Actual Vs Predicted value - Random Forest Regressor')
        fig12.update_layout(
            xaxis_title='Actual value',
            yaxis_title='Predicted value'
        )

        st.plotly_chart(fig12)

        # Evaluating Model Performance
        from sklearn.metrics import mean_absolute_error, r2_score
        st.write('MAE - Random Forest:', mean_absolute_error(y_test, y_pred))
        st.write('MSE - Random Forest:',
                 metrics.mean_squared_error(y_test, y_pred))
        st.write('RMSE - Random Forest:',
                 np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        st.write('R2 - Random Forest:', r2_score(y_test, y_pred))

        st.write("--------------------------------------------")
        st.write("Conclusion of metrics evaluation")
        st.write(f"The R2 is about {round(r2_score(y_test, y_pred),2)} so {round(r2_score(y_test, y_pred)*100,2)}% of the variability in the dependent variable can be explained by the independent variables included in the model.")
        st.write(f"The MAE is about {round(metrics.mean_absolute_error(y_test, y_pred),2)} so on average, the model's predictions are off by about {round(metrics.mean_absolute_error(y_test, y_pred),2)} units of the target variable.")

        ########################################

        st.write("---------------------------------------------")

        st.subheader("Final conclusion and justification of the chosen model.")

        if (r2_score(y_test, y_pred) > lr.score(X, y)) and (mean_absolute_error(y_test, y_pred) < metrics.mean_absolute_error(y_test, predictions)):

            st.write("We choose Random Forest Regressor because in this model, independant variables explain better variance of data. In addition, mean absolute error is lower.")

        else:

            st.write("We choose Linear Regression because in this model, independant variables explain better variance of data. In addition, mean absolute error is lower.")
