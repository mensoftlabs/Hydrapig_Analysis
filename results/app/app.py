import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = r"C:\Users\alvar\OneDrive\Projects\Hydrapig\Results\clustering_data.csv"
data = pd.read_csv(file_path)

# Drop unnecessary columns
data = data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'numero_datos', 'media_mililitros', 
                          'media_segundos_bebedero', 'media_segundos_comedero', 
                          'media_eventos_comedero', 'media_eventos_bebedero'])

# Class definition
class InteractivePigWeightControl:
    def __init__(self, data):
        self.data = data
        self.current_week = 1
        self.expected_gains = []
        self.actual_gains = []
        self.current_weight = None
        self.previous_week_weight = None
        self.initialized = False  # Track if the process is initialized

    def start_process(self, initial_weight):
        """Start the process with the recommendation for the highest cluster in week 1."""
        self.current_weight = initial_weight
        self.previous_week_weight = initial_weight
        self.initialized = True
        initial_recommendation = self.get_recommendation_for_week(week=1, weight_gain_target=None)
        return initial_recommendation

    def get_recommendation_for_week(self, week, weight_gain_target):
        """Get recommendation for the cluster just below the actual gain cluster."""
        week_data = self.data[self.data['Week'] == week]
        
        if weight_gain_target is None:
            # First week: select the cluster with the highest gain
            top_cluster = week_data.loc[week_data['media_ganancia_peso'].idxmax()]
            self.expected_gains.append(round(top_cluster['media_ganancia_peso'], 1))
            return top_cluster.round(1)
        
        else:
            # Find the cluster where the actual gain falls
            real_cluster = week_data.iloc[(week_data['media_ganancia_peso'] - weight_gain_target).abs().argsort()[:1]]
            real_cluster_index = week_data.index.get_loc(real_cluster.index[0])
            
            # Data for the next week
            next_week_data = self.data[self.data['Week'] == week + 1].reset_index(drop=True)
            
            # Check if the lower cluster index is within bounds for the next week
            if real_cluster_index > 0 and real_cluster_index - 1 < len(next_week_data):
                recommended_cluster = next_week_data.iloc[real_cluster_index - 1]
            else:
                recommended_cluster = next_week_data.iloc[0]
            
            self.expected_gains.append(round(recommended_cluster['media_ganancia_peso'], 1))
            return recommended_cluster.round(1)

    def record_weekly_weight(self, new_weight):
        """Register weekly weight and calculate actual gain."""
        if not self.initialized:
            raise ValueError("Initial weight not set. Please start the process with a starting weight.")
        
        actual_gain = round((new_weight - self.previous_week_weight) / 7, 1)
        self.actual_gains.append(actual_gain)
        self.previous_week_weight = new_weight
        self.current_week += 1
        return actual_gain, self.expected_gains[-1]


# Initialize Streamlit interface
st.title("Interactive Pig Weight Control")
st.write("Track weekly weight gains for piglets with clustering-based recommendations.")

# Initialize class instance and session state
if 'interactive_control' not in st.session_state:
    st.session_state.interactive_control = InteractivePigWeightControl(data)
    st.session_state.initialized = False
    st.session_state.initial_weight = None

# Input for weight (initial or weekly)
weight_input = st.number_input(
    "Enter weight (kg):", 
    min_value=0.0, 
    step=0.1, 
    key="weight_input"
)

# Process Start or Update button
if st.button("Calculate"):
    if not st.session_state.initialized:
        # Start process with initial weight
        st.session_state.initial_weight = weight_input
        st.session_state.initialized = True
        recommendation = st.session_state.interactive_control.start_process(st.session_state.initial_weight)
        
        st.write("Initial Recommendation for Week 1:")
        with st.container():
            col1, col2 = st.columns(2)
            # Show table
            with col1:
                st.table(recommendation[['Week', 'Cluster', 'media_ganancia_peso', 'mediana_mililitros', 
                                         'mediana_segundos_bebedero', 'mediana_segundos_comedero', 
                                         'mediana_eventos_bebedero', 'mediana_eventos_comedero']].round(1))
            
            # Initialize plot with dummy data for Week 1
            with col2:
                fig, ax = plt.subplots()
                ax.plot([1], [st.session_state.interactive_control.expected_gains[0]], 'bo-', label="Expected Gain")
                ax.plot([1], [0], 'ro-', label="Actual Gain")  # Placeholder for first week
                ax.set_xlabel("Week")
                ax.set_ylabel("Weight Gain (kg)")
                ax.legend()
                st.pyplot(fig)
    else:
        # Update for subsequent weeks
        actual_gain, expected_gain = st.session_state.interactive_control.record_weekly_weight(weight_input)
        
        st.write(f"**Week {st.session_state.interactive_control.current_week - 1} Results**")
        with st.container():
            col1, col2 = st.columns(2)
            # Show updated table
            recommendation = st.session_state.interactive_control.get_recommendation_for_week(
                st.session_state.interactive_control.current_week - 1, actual_gain
            )
            with col1:
                st.table(recommendation[['Week', 'Cluster', 'media_ganancia_peso', 'mediana_mililitros', 
                                         'mediana_segundos_bebedero', 'mediana_segundos_comedero', 
                                         'mediana_eventos_bebedero', 'mediana_eventos_comedero']].round(1))
            
            # Update plot with actual vs expected gains
            with col2:
                fig, ax = plt.subplots()
                ax.plot(range(1, len(st.session_state.interactive_control.expected_gains) + 1), 
                        st.session_state.interactive_control.expected_gains, 'bo-', label="Expected Gain")
                ax.plot(range(1, len(st.session_state.interactive_control.actual_gains) + 1), 
                        st.session_state.interactive_control.actual_gains, 'ro-', label="Actual Gain")
                ax.set_xlabel("Week")
                ax.set_ylabel("Weight Gain (kg)")
                ax.legend()
                st.pyplot(fig)






