import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import streamlit as st

#st.write('My Data Program')

#Load California DF
california_housing = fetch_california_housing()

#Split DF into Predictor(X) and Target(y) variables
X = pd.DataFrame(california_housing.data, columns = california_housing.feature_names)[['MedInc','HouseAge','AveRooms']]
y = pd.Series(california_housing.target, name = 'MedHouseVal')

#Adding DataFrame X and Series y into a single DataFrame
df = pd.concat([X, y], axis = 1)

#Store X Dataframe in a Variable Called Housing
housing = X

st.set_page_config(page_title = "My Housing Predictions")
st.title("Housing Price Predictions")
st.write("**Use the Sliders On The Left To Adjust the Features**")
st.write("----")

#Sidebar for the User Inputs
with st.sidebar:
    st.title("*Select Your Housing Preferences*")
    #Collect the User Inputs
    user_input = {
        'MedInc': st.slider('Median Income', housing['MedInc'].min(),  housing['MedInc'].max(), housing['MedInc'].median()),
        'HouseAge': st.slider('House Age', housing['HouseAge'].min(), housing['HouseAge'].max(), housing['HouseAge'].median()),
        'AveRooms': st.slider('Average No Of Rooms', housing['AveRooms'].min(), housing['AveRooms'].max(), housing['AveRooms'].median()),
    }

#Put the user input into a Data Frame
user_input_df = pd.DataFrame([user_input])

st.write('\n')
st.markdown("## These are the Features of the House")

#To Display The DataFrame
st.write(user_input_df)

#Initialise the Feature Scaler
scaler = StandardScaler()

#Fit the scaler to both the training data and the user input dataframe
X_scaled = scaler.fit_transform(X)
user_input_scaled = scaler.transform(user_input_df[['MedInc', 'HouseAge', 'AveRooms']])

#Train the model on the scaled data
model = LinearRegression()
model.fit(X_scaled, y)

st.write('\n', '\n')
st.markdown('## The Predictions')
predictions = model.predict(user_input_scaled)[0]
#Ensure Predictions Are Non Negative
predictions = max(predictions, 0)

#Edit the predicted value with a dollar ($) sign and round it up to 1000
st.metric(label = '**This is the *predicted* value**', value = f"${predictions * 1000:.0f}")

#Plotting a Feature Distribution
with st.expander("Feature Distributions"):
    st.subheader("Feature Distributions")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(housing['MedInc'], bins=30, kde=True, ax=axes[0])
    axes[0].set_title('Median Income Distribution')
    sns.histplot(housing['HouseAge'], bins=30, kde=True, ax=axes[1])
    axes[1].set_title('House Age Distribution')
    sns.histplot(housing['AveRooms'], bins=30, kde=True, ax=axes[2])
    axes[2].set_title('Average Rooms Distribution')
    st.pyplot(fig)

#Plotting a PairPlot
with st.expander("PairPlot Of The Features"):
    st.subheader("Pairplot Of Features")
    fig = sns.pairplot(X)
    st.pyplot(fig)