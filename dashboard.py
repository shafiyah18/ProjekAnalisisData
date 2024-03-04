import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
from babel.numbers import format_currency
sns.set(style='dark')

day = pd.read_csv('day.csv')
hour = pd.read_csv('hour.csv')

##Change data types and columns name in the day data
day['dteday'] = pd.to_datetime(day['dteday']) # Converts dteday variable to date data type
day['season'] = day['season'].map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}) # Convert season from integer to text
day['yr'] = day['dteday'].dt.year # Correct the values of yr variable
day['mnth'] = day['dteday'].dt.month_name() # Correct the values of mnth variable 
day['holiday'] = day['workingday'].map({1: 'yes', 0: 'no'}) # Convert "0" to no and "1" to yes
day['weekday'] = day['dteday'].dt.day_name() # Correct the values of weekday variable 
day['workingday'] = day['workingday'].map({1: 'yes', 0: 'no'}) # Convert "0" to no and "1" to yes
day = day.rename(columns = {"dteday":"date", "yr":"year", "mnth":"month", "temp":"temperature", "atemp":"feelingtemperature", "hum":"humidity","cnt":"total"}) # Rename columns

##Change data types and columns name in the hour data
hour['dteday'] = pd.to_datetime(hour['dteday']) # Converts dteday variable to date data type
hour['season'] = hour['season'].map({1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}) # Convert season from integer to text
hour['yr'] = hour['dteday'].dt.year # Correct the values of yr variable 
hour['mnth'] = hour['dteday'].dt.month_name() # Correct the values of mnth variable
hour['holiday'] = hour['workingday'].map({0: 'no', 1: 'yes'}) # Convert "0" to no and "1" to yes
hour['weekday'] = hour['dteday'].dt.day_name() # Correct the values of weekday variable
hour['workingday'] = hour['workingday'].map({0: 'no', 1: 'yes'}) # Convert "0" to no and "1" to yes
hour = hour.rename(columns = {"dteday":"date", "yr":"year", "mnth":"month", "hr":"hour", "temp":"temperature", "atemp":"feelingtemperature", "hum":"humidity","cnt":"total"}) # Rename columns

# Title
st.title('Bike Share Analysis 🚴')
st.write("Bike sharing, also known as bike rental or bike hire, refers to a system where bicycles are made available for shared use by individuals on a short-term basis. These systems typically operate through automated kiosks or mobile apps, allowing users to easily rent a bike for a specific period and return it to any designated docking station within the service area.", align='center')

tab1, tab2, tab3 = st.tabs(["Exploratory", "Visualization", "Inferential"])
 
with tab1:
    st.header("Exploratory")
    # Display total bikeshare users by month
    totalbymonth = day.groupby('month', sort=False).agg({'casual': 'sum', 'registered': 'sum', 'total':'sum'})
    st.write("Total Bike Share Users by Month:")
    st.table(totalbymonth)

    # Display total bikeshare users by year
    totalbyyear = day.groupby('year').agg({'casual': 'sum', 'registered': 'sum', 'total':'sum'}) # Shows casual, registered, and total bikeshare users by year
    st.write("Total Bike Share Users by Year:")
    st.table(totalbyyear)

    col1, col2 = st.columns(2)
    with col1:
      # Display total bikeshare users by holiday
      totalbyholiday = day.groupby('holiday', sort=False).agg({'casual': 'sum', 'registered': 'sum', 'total':'sum'})
      st.write("Total Bike Share Users by Holiday:")
      st.table(totalbyholiday)

    with col2:
      # Display total bikeshare users by workingday
      totalbyworkingday = day.groupby('workingday').agg({'casual': 'sum', 'registered': 'sum', 'total':'sum'})
      st.write("Total Bike Share Users by Workingday:")
      st.table(totalbyworkingday)

    # Display total bikeshare users by weekday
    totalbyweekday = day.groupby('weekday', sort=False).agg({'casual': 'sum', 'registered': 'sum', 'total':'sum'})
    st.write("Total Bike Share Users by Weekday:")
    st.table(totalbyweekday)

    # Display total bikeshare users by weathersit
    totalbyweathersit = day.groupby('weathersit').agg({'casual': 'sum', 'registered': 'sum', 'total':'sum'})
    st.write("Total Bike Share Users by Weathersit:")
    st.table(totalbyweathersit)
    with st.expander("Explanation of Weathersit"):
        st.write(
          """ 1: Clear, Few clouds, Partly cloudy, Partly cloudy;
          2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist;
          3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds;
          4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog """
          
        )

    # Display total bikeshare users by year and month
    totalbyyearmonth = day.groupby(['year', 'month'], sort=False).agg({'casual': 'sum', 'registered': 'sum', 'total':'sum'})
    st.write("Total Bike Share Users by Year and Month:")
    st.table(totalbyyearmonth)

    col1, col2 = st.columns(2)
    with col1:
      # Display mean temperature by season
      meantempbyseason = day.groupby('season').agg({'temperature':'mean'})
      st.write("Mean Temperature by Season:")
      st.table(meantempbyseason)

    with col2:
      # Display mean feeling temperature by season
      meanfeelingtempbyseason = day.groupby('season').agg({'feelingtemperature':'mean'})
      st.write("Mean Feeling Temperature by Season:")
      st.table(meanfeelingtempbyseason)

    # Display total bikeshare users by hour
    totalbyhour = hour.groupby('hour').agg({'casual': 'sum', 'registered': 'sum', 'total':'sum'}) 
    st.write("Total Bike Share Users by Hour:")
    st.table(totalbyhour)

    col1, col2 = st.columns(2)
    with col1:
      # Display total bikeshare users by hour
      meantempbyhour = hour.groupby('hour').agg({'temperature':'mean'})
      st.write("Mean Bike Share Users by Hour:")
      st.table(meantempbyhour)

    with col2:
      # Display total bikeshare users by hour
      meanfeelingtempbyhour = hour.groupby('hour').agg({'feelingtemperature':'mean'})
      st.write("Mean Bike Share Users by Hour:")
      st.table(meanfeelingtempbyhour)

with tab2:
    st.header("Visualization")
    st.subheader("What season do people most often rent bikes?")
    # Grouping by season and aggregating total rides
    seasonly_users_day = day.groupby("season").agg({"casual": "sum", "registered": "sum", "total": "sum"})
    # Plotting the bar chart
    plt.figure(figsize=(10, 5))
    sns.barplot(x="season", y="total", data=seasonly_users_day, order=['Spring', 'Summer', 'Fall', 'Winter'])
    plt.xlabel("Season")
    plt.ylabel("Total Rides")
    plt.title("Count of bikeshare rides by Season")
    st.pyplot(plt)
    with st.expander("Conclusion of the bar chart"):
        st.write(
          """ It can be concluded that the number of people borrowing bikes is highest in the fall season and lowest in the spring season. """  
        )
    st.subheader("Is there a correlation between temperature indicating conditions during a high bike share trip?")
    # Plotting the scatter plot
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x='temperature', y='total', data=day, hue='season')
    plt.xlabel("Temperature (degC)")
    plt.ylabel("Total Rides")
    plt.title("Clusters of bikeshare rides count by season and temperature")
    st.pyplot(plt)
    with st.expander("Conclusion of the scatter plot"):
        st.write(
          """ The number of rides is least at colder temperatures, which occurs during winter, and begins to increase as the temperature increases, which occurs in summer. """  
        )
    st.subheader("What is the proportion of casual users compared to registered users in bike rentals?")
    # Calculate proportions
    proportion_casual = (day['casual'].sum()) / ((day['casual'].sum() + day['registered'].sum()))
    proportion_registered = (day['registered'].sum()) / ((day['casual'].sum() + day['registered'].sum()))
    # Create a pie chart
    labels = ['Casual Users', 'Registered Users']
    sizes = [proportion_casual, proportion_registered]
    colors = ['lightcoral', 'lightskyblue']
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Proportion of Casual Users vs Registered Users')
    st.pyplot(fig)
    with st.expander("Conclusion of the pie chart"):
        st.write(
          """ With this proportion, we can understand the relative usage pattern between regular users and registered users in bicycle rental where registered users use bikeshare more than regular users. """          )
    st.subheader("Is there a correlation between variables such as temperature, humidity, and wind speed and the total number of bike rentals?")
    # Set the deprecation warning option to False
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Calculate correlation matrix
    correlation_matrix = day[['temperature', 'humidity', 'windspeed', 'total']].corr()
    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    plt.title('Correlation Heatmap')
    # Display the heatmap in Streamlit
    st.pyplot(fig)
    with st.expander("Conclusion of the heatmap"):
        st.write(
          """ It can be identified that temperature, humidity, wind speed have the most significant influence on bicycle rentals. In addition, we can assess the strength and direction of these relationships to gain insight into how weather conditions affect bicycle rental patterns. That windspeed and humidity have a negative effect on total bike rentals while temperature has a positive effect on total bike rentals. """          )
    st.subheader("What are the trends in bicycle usage over time, both daily and monthly?")
    # Group by date and calculate total rentals
    daily_rentals = day.groupby('date')['total'].sum()
    # Plot daily rentals trend
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(daily_rentals.index, daily_rentals.values, linestyle='-')
    ax.set_title('Daily Bike Rentals Trend')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Rentals')
    ax.grid(True)
    st.pyplot(fig)
    with st.expander("Conclusion of the line chart"):
        st.write(
          """ The quantity of bike rentals in 2012 surpassed that of 2011. Both years exhibited similar patterns and seasonal variations, characterized by an upsurge in rentals during the middle of the year and a decline at the year's commencement and conclusion. """          )

with tab3:
    st.header("Prediction for 7 days using linier regression")
    # Selecting the required columns
    data = day[['date', 'total']]
    # Splitting the data into training and testing sets
    train_data = data.iloc[:-7]  # Using all data except the last 7 days for training
    test_data = data.iloc[-7:]   # Using the last 7 days for testing
    # Extracting features (dates) for training and testing
    X_train = np.array(train_data.index).reshape(-1, 1)
    X_test = np.array(test_data.index).reshape(-1, 1)
    # Extracting total variable for training
    y_train = train_data['total']
    # Creating and training the linear regression model
    model = np.polyfit(X_train.ravel(), y_train, deg=1)
    # Predicting the values for the next 7 days
    predictions = np.polyval(model, X_test.ravel())
    # Plotting the predictions
    fig, ax = plt.subplots()
    ax.plot(data.index, data['total'], label='Actual')
    ax.plot(X_test, predictions, label='Predicted', linestyle='--', marker='o')
    ax.set_xlabel('Days')
    ax.set_ylabel('Count of Bikes Rented')
    ax.set_title('Predicted Bike Rentals for Next 7 Days')
    ax.legend()
    st.pyplot(fig)
    with st.expander("Conclusion of the prediction"):
        st.write(
          """ From the prediction, it is found that the total bicycle rental continues to increase every day for the next 7 days."""          )



