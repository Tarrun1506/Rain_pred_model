import streamlit as st
import pandas as pd
import pickle as pk

lr_model = pk.load(open("logreg.pkl", "rb"))
svm_model = pk.load(open("svm.pkl", "rb"))

def get_user_input():
    Temperature = st.sidebar.slider('Temperature (°C)', -20.0, 50.0, 25.0, step=0.1)
    Humidity = st.sidebar.slider('Humidity (%)', 0.0, 100.0, 50.0, step=0.1)
    Wind_Speed = st.sidebar.slider('Wind Speed (km/h)', 0.0, 150.0, 10.0, step=0.1)
    Cloud_Cover = st.sidebar.slider('Cloud Cover (%)', 0.0, 100.0, 50.0, step=0.1)
    Pressure = st.sidebar.slider('Pressure (hPa)', 900.0, 1100.0, 1013.0, step=0.1)

    user_data = {
        'Temperature': Temperature,
        'Humidity': Humidity,
        'Wind_Speed': Wind_Speed,
        'Cloud_Cover': Cloud_Cover,
        'Pressure': Pressure,
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

def app():
    st.markdown("# Rain Prediction App")

    user_input = get_user_input()

    st.subheader('User Input:')
    st.write(user_input)

    model_choice = st.sidebar.selectbox('Choose Model', ('Logistic Regression', 'SVM'))

    if st.button('Predict'):
        if model_choice == 'Logistic Regression':
            prediction = lr_model.predict(user_input)
        elif model_choice == 'SVM':
            prediction = svm_model.predict(user_input)

        result = "Rain" if prediction[0] == 1 else "No Rain"
        st.subheader(f'{model_choice} Prediction: {result}')

if __name__ == '__main__':
    app()
