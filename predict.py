"""This create prediction page"""

# Import necessary module
from math import sqrt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error, mean_squared_error

def app(df):
    # Use markdown to give title
    st.markdown("<p style='color:red; font-size: 30px'>This app uses <b>Linear regression</b> to predict the price of a car based on your inputs.</p>", unsafe_allow_html=True)

    # Create a section for user to input data.
    st.header("Select Values:")
    
    # Create sliders.
    car_width = st.slider("Car Width", float(df["carwidth"].min()), float(df["carwidth"].max()))
    engine_size = st.slider("Engine Size", float(df["enginesize"].min()), float(df["enginesize"].max()))
    horse_power = st.slider("Horse Power", float(df["horsepower"].min()), float(df["horsepower"].max()))

    # Creat two radio selection for 0 1 input.
    drivewheel_fwd = st.radio("Is if forward drive wheel car?", ("Yes", "No"))
    car_company_buick = st.radio("Is the car manufactured by Buick?", ("Yes", "No"))

    # Modify radio data.
    if (drivewheel_fwd == "Yes"):
        drivewheel_fwd = 1;
    else:
        drivewheel_fwd = 0;
    
    if (car_company_buick == "Yes"):
        car_company_buick = 1;
    else:
        car_company_buick = 0;

    # Create a list of all input.
    feature_list = [[car_width, engine_size, horse_power, drivewheel_fwd, car_company_buick]]
    
    # Create a button to predict.
    if st.button("Predict"):
        # Get the all values from predict funciton.
        score, pred_price, rsquare_score, mae, msle, rmse = predict(df, feature_list)

        # Display all the values.
        st.success(f"The predicted price of the car: ${int(pred_price):,}")
        st.info(f"Accuracy score of this model is: {score:.2%}")
        st.info(f"R-squared score of this model is: {rsquare_score:.2}")
        st.info(f"Mean absolute error of this model is: {mae:.3f}")
        st.info(f"Mean squared log error of this model is: {msle:.3f}")
        st.info(f"Root mean squared error of this model is: {rmse:.3f}")

@st.cache()
def predict(df, feature_list):
    # Create feature and target variable
    X = df.iloc[:, :-1]
    y = df["price"]

    # Split the data in train test.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create the regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Store score and predicted price in a variable.
    score = model.score(X_train, y_train)
    pred_price = model.predict(feature_list)
    pred_price = pred_price[0]

    # Calculate statical data from the model.
    y_test_pred = model.predict(X_test)
    rsquare_score = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    msle = mean_squared_log_error(y_test, y_test_pred)
    rmse = sqrt(mean_squared_error(y_test, y_test_pred))

    # Return the values.
    return score, pred_price, rsquare_score, mae, msle, rmse

