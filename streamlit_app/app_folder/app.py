from turtle import width
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, RepeatVector
from tensorflow.keras.layers import TimeDistributed, Dense, Flatten
import pickle
import holidays
import pandas as pd
import numpy as np
from pickle import load
import streamlit as st
import folium
from streamlit_folium import st_folium
import altair as alt
from streamlit_option_menu import option_menu
from PIL import Image
import plotly
import plotly.express as px


# App Constants
APP_TITLE = 'Manhattan Taxi Demand Prediction'
APP_SUBTITLE = """
Welcome to Manhattan's Smart Taxi Prediction System! This app helps you plan your taxi rides in Manhattan by predicting how busy different neighborhoods will be.

Here are some tips to enhance your experience:
- 💡 **Check the 'Best Times Today' section** for quieter travel periods.
- 🌧️ **Weather conditions** can affect taxi availability.
- 🕐 **Early morning hours** typically have shorter wait times.
- 📍 **Popular areas** may have longer wait times during peak hours.
"""

def load_model_with_compatibility():
    try:
        # First attempt: Try loading with custom objects and layer config cleanup
        def clean_layer_config(config):
            """Remove incompatible parameters from layer configs"""
            if 'time_major' in config:
                del config['time_major']
            if 'cell' in config and isinstance(config['cell'], dict):
                if 'config' in config['cell']:
                    if 'time_major' in config['cell']['config']:
                        del config['cell']['config']['time_major']
            return config

        # Custom layer loading function
        def custom_layer_from_config(layer_cls, config):
            cleaned_config = clean_layer_config(config)
            return layer_cls.from_config(cleaned_config)

        # Register the custom loading function
        tf.keras.layers.deserialize = lambda config: custom_layer_from_config(
            tf.keras.layers.get(config['class_name']),
            config['config']
        )

        # Try loading the model
        model = tf.keras.models.load_model("CNN_Encoder_Decoder_final_model.h5")
        return model

    except Exception as primary_error:
        try:
            print("Primary loading attempt failed, trying alternative approach...")
            # Fallback: Recreate the model architecture and load weights
            model = Sequential([
                Conv1D(filters=64, kernel_size=2, activation='relu',
                       input_shape=(24, 5)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=32, kernel_size=2, activation='relu'),
                Flatten(),
                RepeatVector(1),
                LSTM(100, return_sequences=True),
                TimeDistributed(Dense(5))
            ])

            # Compile the model with the same configuration
            model.compile(optimizer='adam', loss='mse')

            # Load weights only
            model.load_weights("CNN_Encoder_Decoder_final_model.h5", by_name=True, skip_mismatch=True)
            return model

        except Exception as secondary_error:
            print("Error details:")
            print("Primary error:", str(primary_error))
            print("Secondary error:", str(secondary_error))
            raise Exception("Failed to load model with both approaches. Please check model compatibility.")

def locationsID_Generate_ml_model(dataframe, locationID):
    df = dataframe.copy(deep=True)
    df['LocationID'] = locationID
    df.reset_index(inplace=True, drop=True)
    return df


# 2- This function will be used to transform the the deep learning model prediction to the same format of the machine learning model predction so we will be able to concate them by rows.
def locationsID_Generate_deep_model(dataframe, location):
    df = dataframe.copy(deep=True)
    df['LocationID'] = location
    df['Trips_Forcast'] = df[location]
    df = df[['LocationID', 'datetime', 'Trips_Forcast']]
    df.reset_index(inplace=True, drop=True)
    return df


# 3- Prepare the weather data for the ML model
@st.cache_data
def prepare_weather_data(dataframe):
    '''
    Prepare weather data with correct column names to match model expectations.
    The ML model expects columns with specific names and case sensitivity.
    '''
    dataframe['datetime'] = dataframe['datetime'].astype('datetime64[ns]')
    dataframe['datetime1'] = dataframe['datetime'].dt.strftime('%Y-%m-%d%-H%M')
    # filtering the weather data from 1-May-2022 onward
    dataframe = dataframe[(dataframe['datetime'] >= '2022-05-01')]

    # Create features from datetime column
    dataframe['Year'] = dataframe['datetime'].dt.year
    dataframe['Month'] = dataframe['datetime'].dt.month
    dataframe['DayOfMonth'] = dataframe['datetime'].dt.day
    dataframe['Hour'] = dataframe['datetime'].dt.hour
    dataframe['dayofweek'] = dataframe['datetime'].dt.dayofweek

    # create IsWeekend feature
    dataframe["IsWeekend"] = dataframe["dayofweek"] >= 5
    dataframe['IsWeekend'].replace({True: 1, False: 0}, inplace=True)

    # create date string column
    dataframe['date'] = dataframe['datetime'].apply(lambda x: x.strftime('%d%m%Y'))

    # create holiday column with US holidays
    holiday_list = []
    for holiday in holidays.UnitedStates(years=[2022]).items():
        holiday_list.append(holiday)

    holidays_df = pd.DataFrame(holiday_list, columns=["date", "Holiday"])  # Changed to uppercase Holiday
    holidays_df['Holiday'] = 1  # Changed to uppercase Holiday

    # format date for merging
    holidays_df['date'] = holidays_df['date'].apply(lambda x: x.strftime('%d%m%Y'))

    # join holiday with weather data
    dataframe = dataframe.merge(holidays_df, on='date', how='left')
    dataframe['Holiday'] = dataframe['Holiday'].fillna(0)  # Changed to uppercase Holiday

    # Process locations
    location_list = [263, 262, 261, 249, 246, 244, 243, 239, 238, 234, 233, 232, 231, 230, 229, 224, 211, 209, 202, 194,
                     170, 166, 164, 163, 158, 153, 152, 151, 148, 144, 143, 142, 141, 140, 137, 128, 127, 125, 120, 116,
                     114, 113, 107, 105, 100, 90, 88, 87, 79, 75, 74, 68, 50, 48, 45, 43, 42, 41, 24, 13, 12, 4]

    # create empty dataframe
    df = pd.DataFrame()
    for i in location_list:
        generate_location = locationsID_Generate_ml_model(dataframe, i)
        df = pd.concat([df, generate_location], axis=0)

    # Select features in correct order with correct names
    df_forecast = df[['LocationID', 'Year', 'Month', 'DayOfMonth', 'Hour', 'dayofweek',
                      'temp', 'humidity', 'precip', 'snow', 'windspeed', 'Holiday',  # Note the uppercase Holiday
                      'IsWeekend']]

    # Verify column names match expected
    expected_columns = ['LocationID', 'Year', 'Month', 'DayOfMonth', 'Hour', 'dayofweek',
                        'temp', 'humidity', 'precip', 'snow', 'windspeed', 'Holiday', 'IsWeekend']

    assert all(col in df_forecast.columns for col in expected_columns), "Missing expected columns"

    return df_forecast

# 4- forecast function using ML and DL models and concating the prediction of the two models
@st.cache_data
def data_forcast(dataframe, days_to_forcast):
    # loading the ML and DL models
    with open('ML_model.pkl', 'rb') as f:
        ml_model = pickle.load(f)

    dl_model = load_model_with_compatibility()

    # ================= Forecast using the Machine Learning Model===============================

    # Filter data for the forecast period
    filtered_data = dataframe.loc[
        (dataframe['Month'] == 5) &
        (dataframe['DayOfMonth'] >= 1) &
        (dataframe['DayOfMonth'] < days_to_forcast + 1)
        ].copy()  # Create a copy to avoid modifying original data

    # Make prediction using ML model - use all columns needed for prediction
    ML_forcast = ml_model.predict(filtered_data)

    # Add datetime back to filtered_data if not present
    if 'datetime' not in filtered_data.columns:
        # Create datetime column from components
        filtered_data['datetime'] = pd.to_datetime('2022-05-' +
                                                   filtered_data['DayOfMonth'].astype(str) + ' ' +
                                                   filtered_data['Hour'].astype(str) + ':00:00')

    # Convert predictions to integer and add to filtered data
    filtered_data['Trips_Forcast'] = ML_forcast.astype(int)

    # Select only needed columns
    filtered_data = filtered_data[['LocationID', 'datetime', 'Trips_Forcast']]

    # Convert negative predictions to zero
    filtered_data['Trips_Forcast'] = filtered_data['Trips_Forcast'].clip(lower=0)

    # ================= Forecast using the Deep Learning Model===============================

    # Calculate forecast range
    FORCAST = 24 * days_to_forcast

    # Load and reshape test data
    X_test_mod = np.load('last_24.npy')
    X_test_mod = X_test_mod.reshape((1, 24, 5))

    # Generate predictions
    y_preds = []
    for n in range(FORCAST):
        y_pred = dl_model.predict(X_test_mod, verbose=0)
        X_test_mod = np.append(X_test_mod, y_pred, axis=1)
        X_test_mod = X_test_mod[:, 1:, :]
        y_preds = np.append(y_preds, y_pred)

    y_preds_reshaped = y_preds.reshape(-1, 5)

    # Inverse normalize predictions
    scaler = load(open('scaler1.pkl', 'rb'))
    y_preds_inverse = scaler.inverse_transform(y_preds_reshaped)
    y_preds_inverse = y_preds_inverse.astype(int)

    # Create forecast dataframe
    forcast_df = pd.DataFrame(data=y_preds_inverse, columns=[161, 162, 186, 236, 237])
    future_date = pd.date_range('2022-05-01', periods=FORCAST, freq='H')
    forcast_df['datetime'] = future_date

    # Transform forecast dataframe
    locations = [i for i in forcast_df.columns if i != 'datetime']
    forcast_df_transfrom = pd.DataFrame()
    for i in locations:
        generate_location = locationsID_Generate_deep_model(forcast_df, i)
        forcast_df_transfrom = pd.concat([forcast_df_transfrom, generate_location], axis=0)

    # Clean up predictions
    forcast_df_transfrom['Trips_Forcast'] = forcast_df_transfrom['Trips_Forcast'].clip(lower=0)

    # Combine ML and DL predictions
    combined_forcast = pd.concat([filtered_data, forcast_df_transfrom], axis=0)

    # Add time features
    combined_forcast['DayOfMonth'] = combined_forcast['datetime'].dt.day
    combined_forcast['Hour'] = combined_forcast['datetime'].dt.hour

    # Remove location 105 (not in GeoJSON)
    combined_forcast = combined_forcast[~combined_forcast['LocationID'].isin([105])]

    # Convert LocationID to string
    combined_forcast['LocationID'] = combined_forcast['LocationID'].astype(str)

    return combined_forcast


# =================== Creating the folium map to display the hourly prediction ==================

def display_map(dataframe, day, hour):
    dataframe = dataframe.loc[(dataframe['DayOfMonth'] == day) & (dataframe['Hour'] == hour)]
    dataframe = dataframe.drop(['datetime'], axis=1)

    map = folium.Map(location=[40.7831, -73.9712], zoom_start=11.4, scrollWheelZoom=False, tiles='CartoDB positron')
    choropleth = folium.Choropleth(
        geo_data='manhattan_zones.geojson',
        data=dataframe,
        columns=('LocationID', 'Trips_Forcast'),
        key_on='feature.properties.location_id',
        fill_color="YlOrRd",
        line_opacity=1,
        highlight=True

    )
    choropleth.geojson.add_to(map)

    dataframe = dataframe.set_index('LocationID')

    for feature in choropleth.geojson.data['features']:
        Location_ID = feature['properties']['location_id']
        feature['properties']['Trips_Forcast'] = str(dataframe.loc[Location_ID, 'Trips_Forcast'])

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            fields=['location_id', 'zone', 'Trips_Forcast'],
            aliases=['Location ID : ', 'Area Name : ', 'Forcasted Taxi trips : '])
    )
    st_map = st_folium(map, width=700, height=450)


# ======================== Create time series trend line chart using altair ======================

def line_chart(df):
    brush = alt.selection(type='interval', encodings=['x'])
    base = alt.Chart(df).mark_line().encode(
        x='datetime:T',
        y='Trips_Forcast:Q'
    ).properties(
        width=600,
        height=200)
    upper = base.encode(
        alt.X('datetime:T', scale=alt.Scale(domain=brush)))
    lower = base.properties(
        height=60).add_selection(brush)

    return alt.vconcat(upper, lower)


# ====================== Create bar chart to display the top location ===========================

def bar_plot(dataframe):
    bar = alt.Chart(dataframe).mark_bar().encode(
        x='Trips_Forcast:Q',
        y=alt.Y('Zone:N', sort='-x'),
        color=('Trips_Forcast:Q'),
        tooltip=['Trips_Forcast'])
    return bar


def Hourly_Growth(df, day, hour):
    if hour == 0:

        growth = str(0)
        growth = growth + ' ' + '%'

        return growth
    else:
        end_value = df[(df['DayOfMonth'] == day) & (df['Hour'] == hour)]['Trips_Forcast'].sum()
        start_value = df[(df['DayOfMonth'] == day) & (df['Hour'] == hour - 1)]['Trips_Forcast'].sum()
        growth = str(round((((end_value - start_value) / start_value) * 100), 1))
        growth = growth + ' ' + '%'

        return growth


def Daily_Growth(df, day):
    if day == 1:
        growth = str(0)
        return growth
    else:
        end_value = df[(df['DayOfMonth'] == day)]['Trips_Forcast'].sum()
        start_value = df[(df['DayOfMonth'] == day - 1)]['Trips_Forcast'].sum()
        growth = str(round((((end_value - start_value) / start_value) * 100), 2))
        growth = growth + ' ' + '%'
        return growth

# ======================================== The main streamlit app  ================================

def create_trend_indicator(current_value, previous_value):
    """Create trend indicator with arrow and color"""
    if current_value > previous_value:
        return "↑ Increasing", "green"
    elif current_value < previous_value:
        return "↓ Decreasing", "red"
    return "→ Stable", "gray"

def format_time_period(hour):
    """Convert hour to more readable time period"""
    if 5 <= hour <= 11:
        return "Morning (5 AM - 11 AM)"
    elif 12 <= hour <= 16:
        return "Afternoon (12 PM - 4 PM)"
    elif 17 <= hour <= 21:
        return "Evening (5 PM - 9 PM)"
    return "Night (10 PM - 4 AM)"

def calculate_peak_hours(df, day):
    """Calculate peak hours for a given day"""
    daily_data = df[df['DayOfMonth'] == day].groupby('Hour')['Trips_Forcast'].sum()
    peak_hour = daily_data.idxmax()
    return peak_hour, daily_data[peak_hour]


def predict_future_demand(historical_data, days_ahead=7):
    """Predict demand trends for future days"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(24, 1)),
        tf.keras.layers.Dense(24)
    ])

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(historical_data.reshape(-1, 1))

    predictions = model.predict(scaled_data.reshape(-1, 24, 1))
    return scaler.inverse_transform(predictions)


# Add anomaly detection
def detect_demand_anomalies(data, threshold=2):
    """Detect unusual demand patterns"""
    mean = data['Trips_Forcast'].mean()
    std = data['Trips_Forcast'].std()
    anomalies = data[abs(data['Trips_Forcast'] - mean) > threshold * std]
    return anomalies


# Add competitor analysis section
def analyze_competitors(selected_day, hour):
    """Compare demand with other transportation services"""
    # Simulated competitor data
    competitor_data = {
        'Ride_Share': np.random.normal(1000, 200, 24),
        'Public_Transit': np.random.normal(5000, 1000, 24),
        'Bike_Share': np.random.normal(300, 50, 24)
    }
    return pd.DataFrame(competitor_data)


# Add dynamic pricing suggestions
def suggest_pricing(demand_level, base_price=10):
    """Suggest optimal pricing based on demand"""
    if demand_level > 0.8:
        return base_price * 1.5
    elif demand_level > 0.6:
        return base_price * 1.2
    return base_price


def create_heatmap_legend():
    """Create a custom legend for the demand heatmap"""
    return """
    <div style="position: fixed; bottom: 50px; right: 50px; background: white; padding: 10px; border: 1px solid gray; border-radius: 5px; z-index: 1000;">
        <div style="display: flex; align-items: center; margin: 5px;">
            <div style="width: 20px; height: 20px; background: #fcae91; margin-right: 5px;"></div>
            <span>Medium</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px;">
            <div style="width: 20px; height: 20px; background: #fb6a4a; margin-right: 5px;"></div>
            <span>High</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px;">
            <div style="width: 20px; height: 20px; background: #cb181d; margin-right: 5px;"></div>
            <span>Very High</span>
        </div>
    </div>
    """

def main():
    # Configure page settings
    st.set_page_config(APP_TITLE, layout="wide")

    # Enhanced custom CSS
    st.markdown("""
        <style>
        /* Modern color palette */
        :root {
            --primary-color: #2C3E50;
            --secondary-color: #34495E;
            --accent-color: #3498DB;
            --success-color: #27AE60;
            --warning-color: #F39C12;
            --danger-color: #E74C3C;
            --light-bg: #ECF0F1;
            --dark-bg: #2C3E50;
            --text-light: #ECF0F1;
            --text-dark: #2C3E50;
        }

        /* Base styles */
        .stApp {
            background-color: #f8f9fa;
        }

        /* Typography */
        .big-font {
            font-size: 28px !important;
            font-weight: bold;
            color: var(--primary-color);
            margin-bottom: 20px;
        }

        /* Enhanced metric cards */
        .metric-card {
            background: linear-gradient(135deg, #FFF, var(--light-bg));
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 25px;
            border: 1px solid rgba(0,0,0,0.05);
            transition: transform 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-5px);
        }

        .metric-card h4 {
            color: var(--primary-color);
            font-size: 18px;
            margin-bottom: 15px;
        }

        /* Improved insight cards */
        .insight-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid var(--accent-color);
            margin-bottom: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }

        /* Enhanced trend indicators */
        .trend-up {
            color: var(--success-color);
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .trend-down {
            color: var(--danger-color);
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .trend-stable {
            color: var(--warning-color);
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        /* Map legend */
        .map-legend {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-top: 15px;
        }

        /* Dashboard sections */
        .dashboard-section {
            background-color: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.05);
        }

        /* Chart containers */
        .chart-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }

        </style>
    """, unsafe_allow_html=True)

    # Enhanced sidebar navigation
    with st.sidebar:
        selected = option_menu(
            menu_title='Navigation',
            options=['Dashboard', 'Detailed Analysis'],
            icons=['speedometer2', 'graph-up', 'info-circle'],
            default_index=0,
            styles={
                "container": {"padding": "15px", "background-color": "#f8f9fa"},
                "icon": {"color": "#3498DB", "font-size": "20px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#3498DB"},
            }
        )

    if selected == 'Dashboard':
        # Header Section
        st.markdown(f'<h1 style="color: var(--primary-color);">📊 {APP_TITLE}</h1>', unsafe_allow_html=True)
        st.markdown('<p class="big-font">Real-time Taxi Demand Insights</p>', unsafe_allow_html=True)
        st.caption(APP_SUBTITLE)

        # Data Loading and Processing
        df_temp = pd.read_csv('new york_weather.csv',
                            usecols=['datetime', 'temp', 'humidity', 'precip', 'snow', 'windspeed'])
        df_forecast = prepare_weather_data(df_temp)

        # Forecast Settings
        st.sidebar.markdown('<h3 style="color: var(--primary-color);">📅 Forecast Settings</h3>', unsafe_allow_html=True)
        forecast_days = st.sidebar.selectbox(
            'Select Forecast Period',
            [3, 7, 14],
            format_func=lambda x: f'{x} Days Forecast'
        )

        # Generate Forecast
        combined_forecast = data_forcast(df_forecast, forecast_days)

        # Date and Time Selection
        col1, col2 = st.sidebar.columns(2)
        with col1:
            day_list = list(combined_forecast['DayOfMonth'].unique())
            selected_day = st.selectbox('Select Date', day_list,
                                    format_func=lambda x: f'{x}-May-2022')
        with col2:
            hour_list = list(combined_forecast['Hour'].unique())
            selected_hour = st.selectbox('Select Hour', hour_list,
                                     format_func=lambda x: f'{x:02d}:00')

        # Main Dashboard Layout
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: var(--primary-color);">🗺️ Real-time Demand Heatmap</h3>', unsafe_allow_html=True)
            display_map(combined_forecast, selected_day, selected_hour)

        with col2:
            # Key Metrics
            st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: var(--primary-color);">📊 Key Metrics</h3>', unsafe_allow_html=True)

            # Calculate metrics
            total_daily_trips = combined_forecast[
                combined_forecast['DayOfMonth'] == selected_day]['Trips_Forcast'].sum()
            total_hourly_trips = combined_forecast[
                (combined_forecast['DayOfMonth'] == selected_day) &
                (combined_forecast['Hour'] == selected_hour)]['Trips_Forcast'].sum()

            daily_growth = Daily_Growth(combined_forecast, selected_day)
            hourly_growth = Hourly_Growth(combined_forecast, selected_day, selected_hour)

            # Enhanced metric cards
            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Daily Trips</h4>
                                <p class="big-font">{format(int(total_daily_trips), ',')}</p>
                                <p class="trend-up">
                                    <span>↑</span> {daily_growth} vs Previous Day
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Hourly Trips</h4>
                                <p class="big-font">{format(int(total_hourly_trips), ',')}</p>
                                <p class="trend-up">
                                    <span>↑</span> {hourly_growth} vs Previous Hour
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Add this to Quick Insights
            # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<h4 style="color: var(--primary-color);">🎯 Best Times Today</h4>', unsafe_allow_html=True)

            # Get hourly demand for the day
            hourly_demand = combined_forecast[
                combined_forecast['DayOfMonth'] == selected_day
                ].groupby('Hour')['Trips_Forcast'].sum()

            # Find quiet hours (lowest demand periods)
            quiet_hours = hourly_demand.nsmallest(3)

            st.markdown("""
                            <p><strong>Recommended travel times:</strong></p>
                        """, unsafe_allow_html=True)

            for hour, demand in quiet_hours.items():
                # Format time in 12-hour format
                time_str = f"{hour:02d}:00"
                am_pm = "AM" if hour < 12 else "PM"
                hour_12 = hour if hour <= 12 else hour - 12
                hour_12 = 12 if hour_12 == 0 else hour_12

                st.markdown(f"""
                                <p>🕐 {hour_12}:00 {am_pm}</p>
                            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)


        with col3:
            # Quick Insights
            st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: var(--primary-color);">💡 Quick Insights</h3>', unsafe_allow_html=True)

            # Calculate insights
            hourly_data = combined_forecast[
                combined_forecast['DayOfMonth'] == selected_day].groupby('Hour')['Trips_Forcast'].sum()
            peak_hour = hourly_data.idxmax()
            peak_demand = hourly_data.max()

            # Peak Hour card
            st.markdown(f"""
                <div class="insight-card" style="background-color: white; border-radius: 8px; padding: 16px; margin: 8px 0; border-left: 4px solid #3182ce; box-shadow: 0 1px 3px rgba(0,0,0,0.12);">
                    <h4 style="color: var(--primary-color); margin-bottom: 8px;">Peak Hour</h4>
                    <p>{peak_hour:02d}:00 ({format(int(peak_demand), ',')} trips)</p>
                </div>
            """, unsafe_allow_html=True)

            # Weather insights card
            weather_data = df_temp[df_temp['datetime'].str.contains(f'2022-05-{selected_day:02d}')].iloc[selected_hour]
            st.markdown(f"""
                <div class="insight-card" style="background-color: white; border-radius: 8px; padding: 16px; margin: 8px 0; border-left: 4px solid #3182ce; box-shadow: 0 1px 3px rgba(0,0,0,0.12);">
                    <h4 style="color: var(--primary-color); margin-bottom: 8px;">Weather Conditions</h4>
                    <p>Temperature: {weather_data['temp']}°C</p>
                    <p>Precipitation: {weather_data['precip']}mm</p>
                </div>
            """, unsafe_allow_html=True)

            # Current Status card
            current_demand = combined_forecast[
                (combined_forecast['DayOfMonth'] == selected_day) &
                (combined_forecast['Hour'] == selected_hour)
                ]['Trips_Forcast'].sum()

            def get_demand_status(demand):
                if demand > 5000:
                    return "Very Busy", "#ff4b4b"  # Red
                elif demand > 3000:
                    return "Busy", "#ffa500"  # Orange
                elif demand > 1000:
                    return "Moderate", "#ffeb3b"  # Yellow
                else:
                    return "Quiet", "#00ff00"  # Green

            status, color = get_demand_status(current_demand)

            st.markdown(f"""
                <div class="insight-card" style="background-color: white; border-radius: 8px; padding: 16px; margin: 8px 0; border-left: 4px solid #3182ce; box-shadow: 0 1px 3px rgba(0,0,0,0.12);">
                    <h4 style="color: var(--primary-color); margin-bottom: 8px;">🚦 Current Status</h4>
                    <div style="text-align: center;">
                        <h2 style="color: {color}; font-size: 24px; margin: 8px 0;">{status}</h2>
                        <p>Best to {
            'wait if possible' if status == 'Very Busy'
            else 'expect delays' if status == 'Busy'
            else 'go now' if status == 'Moderate'
            else 'go now - minimal wait'
            }</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Estimated Wait Times card
            def estimate_wait_time(demand):
                if demand > 5000:
                    return "15+ minutes"
                elif demand > 3000:
                    return "10-15 minutes"
                elif demand > 1000:
                    return "5-10 minutes"
                else:
                    return "Less than 5 minutes"

            wait_time = estimate_wait_time(current_demand)

            st.markdown(f"""
                <div class="insight-card" style="background-color: white; border-radius: 8px; padding: 16px; margin: 8px 0; border-left: 4px solid #3182ce; box-shadow: 0 1px 3px rgba(0,0,0,0.12);">
                    <h4 style="color: var(--primary-color); margin-bottom: 8px;">⏱️ Estimated Wait Times</h4>
                    <div style="text-align: center;">
                        <h3 style="color: var(--primary-color); margin: 8px 0;">{wait_time}</h3>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)



        # Trend Analysis Section
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: var(--primary-color);">📈 Demand Trends</h3>', unsafe_allow_html=True)

        # Time series chart
        hourly_trend = combined_forecast[
            combined_forecast['DayOfMonth'] == selected_day].groupby('Hour')['Trips_Forcast'].sum().reset_index()

        fig = px.line(hourly_trend, x='Hour', y='Trips_Forcast',
                    title='Hourly Demand Pattern',
                    labels={'Trips_Forcast': 'Predicted Trips', 'Hour': 'Hour of Day'})
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_x=0.5,
            title_font_size=20
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Top Locations Analysis
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: var(--primary-color);">📍 Top Locations</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        # Add this inside the Quick Insights section (col3)
        st.markdown("""
            <div class="insight-card">
                <h4 style="color: var(--primary-color);">Daily Peak Hours</h4>
        """, unsafe_allow_html=True)

        # Get hourly demand patterns
        daily_data = combined_forecast[
            combined_forecast['DayOfMonth'] == selected_day
            ].groupby('Hour')['Trips_Forcast'].sum()

        # Morning peak (6-10 AM)
        morning_peak = daily_data[6:11].max()
        morning_peak_hour = daily_data[6:11].idxmax()

        # Evening peak (4-8 PM)
        evening_peak = daily_data[16:21].max()
        evening_peak_hour = daily_data[16:21].idxmax()

        st.markdown(f"""
            <p>🌅 Morning Peak: {morning_peak_hour:02d}:00 ({format(int(morning_peak), ',')} trips)</p>
            <p>🌆 Evening Peak: {evening_peak_hour:02d}:00 ({format(int(evening_peak), ',')} trips)</p>
        """, unsafe_allow_html=True)

        # Add this after the Weather Conditions card in Quick Insights
        weather_data = df_temp[df_temp['datetime'].str.contains(f'2022-05-{selected_day:02d}')].iloc[selected_hour]

        # Define weather impact conditions
        def get_weather_impact():
            impact = "Low"
            color = "green"
            if weather_data['precip'] > 5:
                impact = "High"
                color = "red"
            elif weather_data['precip'] > 2:
                impact = "Medium"
                color = "orange"

            return impact, color

        impact, color = get_weather_impact()

        st.markdown(f"""
            <div class="insight-card">
                <h4 style="color: var(--primary-color);">Weather Impact Alert</h4>
                <p>Expected Impact: <span style="color: {color}; font-weight: bold;">{impact}</span></p>
                <p>Precipitation: {weather_data['precip']}mm</p>
                <p>Wind Speed: {weather_data['windspeed']} km/h</p>
            </div>
        """, unsafe_allow_html=True)

        # Add this after the Trend Analysis section
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: var(--primary-color);">⏰ Demand Distribution</h3>', unsafe_allow_html=True)

        # Calculate demand by time period
        daily_demand = combined_forecast[combined_forecast['DayOfMonth'] == selected_day]
        time_periods = {
            'Morning (6-10)': daily_demand[daily_demand['Hour'].between(6, 10)]['Trips_Forcast'].sum(),
            'Midday (11-15)': daily_demand[daily_demand['Hour'].between(11, 15)]['Trips_Forcast'].sum(),
            'Evening (16-20)': daily_demand[daily_demand['Hour'].between(16, 20)]['Trips_Forcast'].sum(),
            'Night (21-5)': daily_demand[
                daily_demand['Hour'].isin(list(range(21, 24)) + list(range(0, 6)))
            ]['Trips_Forcast'].sum()
        }

        # Create pie chart using plotly
        fig = px.pie(
            values=list(time_periods.values()),
            names=list(time_periods.keys()),
            title='Daily Demand Distribution by Time Period'
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_x=0.5,
            title_font_size=20
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        with col1:
            st.markdown('<h4 style="color: var(--primary-color);">Top 10 Pickup Locations (Hourly)</h4>', unsafe_allow_html=True)
            location_lookup = pd.read_csv('taxi_Zone_lookup.csv',
                                        usecols=['LocationID', 'Zone'])
            location_lookup['LocationID'] = location_lookup['LocationID'].astype(str)

            # Hourly top locations
            hourly_locations = combined_forecast[
                (combined_forecast['DayOfMonth'] == selected_day) &
                (combined_forecast['Hour'] == selected_hour)]
            hourly_locations = hourly_locations.merge(
                location_lookup, on='LocationID', how='left')
            hourly_locations = hourly_locations.nlargest(10, "Trips_Forcast")

            # Hourly locations chart
            hourly_chart = bar_plot(hourly_locations)
            st.altair_chart(hourly_chart, use_container_width=True)



        with col2:
            st.markdown('<h4 style="color: var(--primary-color);">Top 10 Pickup Locations (Daily)</h4>',
                        unsafe_allow_html=True)

            # Daily top locations
            daily_locations = combined_forecast[combined_forecast['DayOfMonth'] == selected_day]
            daily_locations = daily_locations.groupby('LocationID')[['Trips_Forcast']].sum()
            daily_locations.reset_index(inplace=True)
            daily_locations['LocationID'] = daily_locations['LocationID'].astype(str)
            daily_locations = daily_locations.merge(location_lookup, on='LocationID', how='left')
            daily_locations = daily_locations.nlargest(10, "Trips_Forcast")

            daily_chart = bar_plot(daily_locations)
            st.altair_chart(daily_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    elif selected == 'Detailed Analysis':
        st.markdown('<h1 style="color: var(--primary-color);">📊 Detailed Analysis</h1>', unsafe_allow_html=True)

        # Load and process data
        df_temp = pd.read_csv('new york_weather.csv',
                              usecols=['datetime', 'temp', 'humidity', 'precip', 'snow', 'windspeed'])
        df_forecast = prepare_weather_data(df_temp)

        # Enhanced forecast period selection
        st.sidebar.markdown('<h3 style="color: var(--primary-color);">📅 Analysis Settings</h3>', unsafe_allow_html=True)
        forecast_days = st.sidebar.selectbox(
            'Select Forecast Period',
            [3, 7, 14],
            format_func=lambda x: f'{x} Days Forecast'
        )

        combined_forecast = data_forcast(df_forecast, forecast_days)

        # Time Series Analysis Section
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: var(--primary-color);">📈 Time Series Analysis</h3>', unsafe_allow_html=True)

        # Enhanced date range selector
        day_range = st.slider(
            'Select Date Range',
            min_value=min(combined_forecast['DayOfMonth']),
            max_value=max(combined_forecast['DayOfMonth']),
            value=(min(combined_forecast['DayOfMonth']), min(combined_forecast['DayOfMonth']) + 2)
        )

        # Filter and prepare data
        filtered_data = combined_forecast[
            (combined_forecast['DayOfMonth'] >= day_range[0]) &
            (combined_forecast['DayOfMonth'] <= day_range[1])
            ]

        # Enhanced time series chart
        daily_demand = filtered_data.groupby(['DayOfMonth', 'Hour'])['Trips_Forcast'].sum().reset_index()
        daily_demand['datetime'] = pd.to_datetime('2022-05-' + daily_demand['DayOfMonth'].astype(str) + ' ' +
                                                  daily_demand['Hour'].astype(str) + ':00:00')

        fig = px.line(daily_demand, x='datetime', y='Trips_Forcast',
                      title='Hourly Demand Trend',
                      labels={'Trips_Forcast': 'Predicted Trips', 'datetime': 'Date and Time'})
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_x=0.5,
            title_font_size=20
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Demand Pattern Analysis Section
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: var(--primary-color);">🔍 Demand Pattern Analysis</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Enhanced hourly pattern chart
            hourly_pattern = filtered_data.groupby('Hour')['Trips_Forcast'].mean().reset_index()
            fig_hourly = px.bar(hourly_pattern, x='Hour', y='Trips_Forcast',
                                title='Average Hourly Demand Pattern',
                                labels={'Trips_Forcast': 'Average Trips', 'Hour': 'Hour of Day'})
            fig_hourly.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                title_x=0.5
            )
            st.plotly_chart(fig_hourly, use_container_width=True)

        with col2:
            # Enhanced daily pattern chart
            daily_pattern = filtered_data.groupby('DayOfMonth')['Trips_Forcast'].sum().reset_index()
            fig_daily = px.bar(daily_pattern, x='DayOfMonth', y='Trips_Forcast',
                               title='Daily Demand Pattern',
                               labels={'Trips_Forcast': 'Total Trips', 'DayOfMonth': 'Day of Month'})
            fig_daily.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                title_x=0.5
            )
            st.plotly_chart(fig_daily, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Location Analysis Section
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: var(--primary-color);">📍 Location-based Analysis</h3>', unsafe_allow_html=True)

        # Load location data
        location_lookup = pd.read_csv('taxi_Zone_lookup.csv', usecols=['LocationID', 'Zone'])
        location_lookup['LocationID'] = location_lookup['LocationID'].astype(str)

        # Enhanced location selector
        selected_locations = st.multiselect(
            'Select Locations to Compare',
            options=location_lookup['Zone'].unique(),
            default=location_lookup['Zone'].iloc[:3]
        )

        if selected_locations:
            selected_ids = location_lookup[location_lookup['Zone'].isin(selected_locations)]['LocationID']
            location_data = filtered_data[filtered_data['LocationID'].isin(selected_ids)]
            location_data = location_data.merge(location_lookup, on='LocationID', how='left')

            # Enhanced location comparison chart
            fig_locations = px.line(
                location_data,
                x='Hour',
                y='Trips_Forcast',
                color='Zone',
                title='Demand Comparison by Location',
                labels={'Trips_Forcast': 'Predicted Trips', 'Hour': 'Hour of Day'}
            )
            fig_locations.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                title_x=0.5
            )
            st.plotly_chart(fig_locations, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Weather Impact Analysis Section
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: var(--primary-color);">🌤️ Weather Impact Analysis</h3>', unsafe_allow_html=True)

        try:
            # First ensure datetime is in the right format and extract hour
            df_temp['datetime'] = pd.to_datetime(df_temp['datetime'])
            df_temp['hour'] = df_temp['datetime'].dt.hour
            df_temp['day_of_month'] = df_temp['datetime'].dt.day

            # Prepare weather impact data
            weather_impact = filtered_data.merge(
                df_temp,
                left_on=['DayOfMonth', 'Hour'],
                right_on=['day_of_month', 'hour'],
                how='left'
            )

            col1, col2 = st.columns(2)

            with col1:
                # Enhanced temperature impact chart
                fig_temp = px.scatter(
                    weather_impact,
                    x='temp',
                    y='Trips_Forcast',
                    title='Temperature vs Demand',
                    labels={'temp': 'Temperature (°C)', 'Trips_Forcast': 'Predicted Trips'},
                    trendline="lowess"  # Added trendline for better visualization
                )
                fig_temp.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    title_x=0.5,
                    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
                )
                st.plotly_chart(fig_temp, use_container_width=True)

            with col2:
                # Enhanced precipitation impact chart
                fig_precip = px.scatter(
                    weather_impact,
                    x='precip',
                    y='Trips_Forcast',
                    title='Precipitation vs Demand',
                    labels={'precip': 'Precipitation (mm)', 'Trips_Forcast': 'Predicted Trips'},
                    trendline="lowess"  # Added trendline for better visualization
                )
                fig_precip.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    title_x=0.5,
                    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
                )
                st.plotly_chart(fig_precip, use_container_width=True)

        except Exception as e:
            st.error(f"""
                Error in Weather Impact Analysis: {str(e)}
                Please check that your data contains the required columns:
                - filtered_data should have: DayOfMonth, Hour, Trips_Forcast
                - df_temp should have: datetime, temp, precip
            """)

        st.markdown('</div>', unsafe_allow_html=True)

    else:  # About page
        st.markdown('<h1 style="color: var(--primary-color);">ℹ️ About</h1>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()