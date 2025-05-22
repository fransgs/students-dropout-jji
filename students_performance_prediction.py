import datetime
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
import io
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
APP_TITLE = "🏛️ Jaya Jaya Institute Student Outcome Predictor"
APP_ICON = "🏛️"
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# --- Configuration & Constants ---
# File Paths
MODEL_PATH = Path("rf_model.joblib")
BASE_DATA_FOR_SCALING_PATH = Path("data_filtered.csv") # Used to FIT the scaler
TEMPLATE_EXCEL_PATH = Path("data_template.xlsx")

# Mappings
GENDER_MAPPING: Dict[str, int] = {'Male': 1, 'Female': 0}
MARITAL_STATUS_MAPPING: Dict[str, int] = {
    'Single': 1, 'Married': 2, 'Widower': 3, 'Divorced': 4,
    'Facto Union': 5, 'Legally Seperated': 6
}
APPLICATION_MODE_MAPPING: Dict[str, int] = {
    '1st Phase - General Contingent': 1,
    '1st Phase - Special Contingent (Azores Island)': 5,
    '1st Phase - Special Contingent (Madeira Island)': 16,
    '2nd Phase - General Contingent': 17,
    '3rd Phase - General Contingent': 18,
    'Ordinance No. 612/93': 2,
    'Ordinance No. 854-B/99': 10,
    'Ordinance No. 533-A/99, Item B2 (Different Plan)': 26,
    'Ordinance No. 533-A/99, Item B3 (Other Institution)': 27,
    'International Student (Bachelor)': 15,
    'Over 23 Years Old': 39,
    'Transfer': 42,
    'Change of Course': 43,
    'Holders of Other Higher Courses': 7,
    'Short Cycle Diploma Holders': 53,
    'Technological Specialization Diploma Holders': 44,
    'Change of Institution/Course': 51,
    'Change of Institution/Course (International)': 57,
}

# --- Caching & Resource Loading ---
@st.cache_resource
def load_model_and_fit_scaler(
    model_path: Path, base_data_csv_path: Path
) -> Tuple[Optional[Any], Optional[StandardScaler], Optional[List[str]]]:
    """
    Loads the pre-trained model.
    Loads the base data CSV, fits a StandardScaler on its features,
    and returns the model, fitted scaler, and feature column names.
    """
    model = None
    scaler = None
    feature_columns = None

    # Load Model
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"CRITICAL ERROR: Model file '{model_path}' not found. Application cannot make predictions.")
        return None, None, None
    except Exception as e:
        st.error(f"CRITICAL ERROR: Could not load model from '{model_path}'. Error: {e}")
        return None, None, None

    # Load base data and fit Scaler
    try:
        base_df = pd.read_csv(base_data_csv_path)
        if 'Status' in base_df.columns: # Assuming 'Status' is the target and not a feature
            features_df = base_df.drop(columns=['Status'], axis=1)
        else:
            features_df = base_df.copy()
        
        # Ensure all feature columns are numeric for scaler
        non_numeric_cols = [col for col in features_df.columns if not pd.api.types.is_numeric_dtype(features_df[col])]
        if non_numeric_cols:
            st.error(
                f"ERROR: Non-numeric columns found in '{base_data_csv_path}' used for scaling: {', '.join(non_numeric_cols)}. "
                "All feature columns in this file (excluding 'Status') must be numeric to fit the scaler."
            )
            return model, None, None # Return model but no scaler

        scaler = StandardScaler()
        scaler.fit(features_df)
        feature_columns = features_df.columns.tolist()
        # st.success(f"Scaler successfully fitted using data from '{base_data_csv_path}'.") # Optional: can be noisy

    except FileNotFoundError:
        st.error(f"ERROR: Base data file '{base_data_csv_path}' for scaler fitting not found. Scaler cannot be fitted.")
        return model, None, None
    except pd.errors.EmptyDataError:
        st.error(f"ERROR: Base data file '{base_data_csv_path}' is empty. Scaler cannot be fitted.")
        return model, None, None
    except Exception as e:
        st.error(f"ERROR during scaler fitting with '{base_data_csv_path}': {e}")
        return model, None, None

    return model, scaler, feature_columns

MODEL, SCALER, DF_EXPECTED_COLUMNS = load_model_and_fit_scaler(MODEL_PATH, BASE_DATA_FOR_SCALING_PATH)

# Fallback if DF_EXPECTED_COLUMNS could not be determined (e.g., scaler fitting failed)
if DF_EXPECTED_COLUMNS is None:
    st.warning(
        "WARNING: Could not determine expected feature columns from base data file "
        f"('{BASE_DATA_FOR_SCALING_PATH}'). Using a default list. This may lead to errors or inaccuracies "
        "if it doesn't match your model's training features."
    )
    DF_EXPECTED_COLUMNS = [
        'Marital_status', 'Application_mode', 'Previous_qualification_grade',
        'Admission_grade', 'Displaced', 'Debtor', 'Tuition_fees_up_to_date',
        'Gender', 'Scholarship_holder', 'Age_at_enrollment',
        'Curricular_units_1st_sem_enrolled',
        'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
        'Curricular_units_2nd_sem_enrolled',
        'Curricular_units_2nd_sem_evaluations',
        'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
        'Curricular_units_2nd_sem_without_evaluations'
    ]

# --- Helper Functions ---
def preprocess_data(input_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Preprocesses the input DataFrame:
    1. Ensures columns are in the correct order as defined by DF_EXPECTED_COLUMNS.
    2. Applies the pre-fitted scaler (if available).
    """
    if SCALER is None:
        st.error("Scaler is not available (likely failed to fit). Cannot preprocess data with scaling.")
        # Optionally, you could proceed without scaling if you want to allow predictions,
        # but it's better to highlight the issue.
        return None # Or return input_df if you want to allow unscaled prediction
    if not DF_EXPECTED_COLUMNS: # Check if list is empty or None
        st.error("List of expected feature columns is not available. Cannot ensure data integrity for preprocessing.")
        return None
        
    try:
        # Ensure all expected columns are present, fill missing with 0.
        # This is a simple imputation. More sophisticated might be needed.
        processed_df = pd.DataFrame(columns=DF_EXPECTED_COLUMNS)
        for col in DF_EXPECTED_COLUMNS:
            if col in input_df.columns:
                processed_df[col] = input_df[col]
            else:
                processed_df[col] = 0 # Fill missing expected columns with 0
        
        # Reorder to be absolutely sure, though constructing it column by column helps
        processed_df = processed_df[DF_EXPECTED_COLUMNS]
        
        # Ensure all data is numeric before scaling
        non_numeric_input_cols = [col for col in processed_df.columns if not pd.api.types.is_numeric_dtype(processed_df[col])]
        if non_numeric_input_cols:
            st.error(
                f"DATA ERROR: Non-numeric data found in input for columns: {', '.join(non_numeric_input_cols)}. "
                "All features must be numbers after mapping and before scaling."
            )
            return None

        scaled_data = SCALER.transform(processed_df)
        return pd.DataFrame(scaled_data, columns=DF_EXPECTED_COLUMNS, index=processed_df.index) # Preserve index
    except ValueError as ve:
        st.error(
            f"ERROR during data scaling: {ve}. This might be due to a mismatch in the number of features "
            "or data types expected by the scaler. Ensure input data matches the structure of "
            f"'{BASE_DATA_FOR_SCALING_PATH}' (after mappings)."
        )
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during data preprocessing: {e}")
        return None

def predict_status(processed_df: pd.DataFrame) -> Optional[List[str]]:
    """Makes predictions using the loaded model."""
    if MODEL is None:
        st.error("Model not loaded. Cannot make predictions.")
        return None
    try:
        predictions = MODEL.predict(processed_df)
        return ['Graduate' if pred == 1 else 'Dropout' for pred in predictions]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

def get_color_for_status(status: str) -> str:
    return 'color: green' if status == 'Graduate' else 'color: red'

def display_prediction_result(output_status: str):
    if output_status == 'Graduate':
        st.success(f"Predicted Student Status: **{output_status}** 🎉🎓")
        st.balloons()
    else:
        st.error(f"Predicted Student Status: **{output_status}**  😭😔💔")


# --- UI Components ---
def single_prediction_tab():
    st.header("🔮 Single Student Prediction")
    st.markdown("Enter the student's details below to predict their academic outcome.")

    with st.form(key="single_student_form"):
        st.subheader("👤 Personal Information")
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            gender_str = st.radio('Gender', options=list(GENDER_MAPPING.keys()), index=0, horizontal=True, help='The gender of the student')
        with col2:
            age = st.number_input('Age at Enrollment', min_value=17, max_value=70, value=20, help='Student age at enrollment')
        with col3:
            marital_status_str = st.selectbox('Marital Status', options=list(MARITAL_STATUS_MAPPING.keys()), index=0, help='Marital status')

        st.subheader("📝 Application & Academic Background")
        col_app, col_prev_grade, col_adm_grade = st.columns([3, 2, 2])
        with col_app:
            application_mode_str = st.selectbox('Application Mode', options=list(APPLICATION_MODE_MAPPING.keys()), index=0, help='Application method')
        with col_prev_grade:
            prev_qualification_grade = st.number_input('Previous Qualification Grade', min_value=0, max_value=200, value=120, help='Grade of previous qualification (0-200)')
        with col_adm_grade:
            admission_grade = st.number_input('Admission Grade', min_value=0, max_value=200, value=120, help="Admission grade (0-200)")

        st.subheader("💰 Financial & Other Information")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            scholarship_holder = 1 if st.checkbox('Scholarship Holder?', value=False, help='Is the student a scholarship holder?') else 0
        with c2:
            tuition_fees_up_to_date = 1 if st.checkbox('Tuition Up-to-Date?', value=True, help="Are tuition fees up to date?") else 0
        with c3:
            displaced = 1 if st.checkbox('Displaced?', value=False, help='Is the student displaced?') else 0
        with c4:
            debtor = 1 if st.checkbox('Debtor?', value=False, help='Is the student a debtor?') else 0
        
        st.subheader("📚 Curricular Units Performance")
        st.markdown("_(Enter values for 1st and 2nd semester)_")
        
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        with row1_col1:
            curricular_units_1st_sem_enrolled = st.number_input('1st Sem Enrolled', min_value=0, max_value=26, value=5, help='Units enrolled in 1st semester')
            curricular_units_1st_sem_approved = st.number_input('1st Sem Approved', min_value=0, max_value=26, value=5, help='Units approved in 1st semester')
            curricular_units_1st_sem_grade = st.number_input('1st Sem Avg. Grade', min_value=0.0, max_value=20.0, value=10.0, step=0.1, format="%.1f", help='Average grade in 1st semester (0-20)')
        
        with row1_col2:
            curricular_units_2nd_sem_enrolled = st.number_input('2nd Sem Enrolled', min_value=0, max_value=23, value=5, help='Units enrolled in 2nd semester')
            curricular_units_2nd_sem_approved = st.number_input('2nd Sem Approved', min_value=0, max_value=20, value=5, help='Units approved in 2nd semester')
            curricular_units_2nd_sem_grade = st.number_input('2nd Sem Avg. Grade', min_value=0.0, max_value=20.0, value=10.0, step=0.1, format="%.1f", help='Average grade in 2nd semester (0-20)')

        with row1_col3:
            curricular_units_2nd_sem_evaluations = st.number_input('2nd Sem Evaluations', min_value=0, max_value=33, value=5, help='Number of evaluations in 2nd semester')
            curricular_units_2nd_sem_without_evaluations = st.number_input('2nd Sem w/o Evaluations', min_value=0, max_value=12, value=0, help='Units without evaluations in 2nd semester')

        submit_button = st.form_submit_button(label='✨ Predict Student Outcome', use_container_width=True)

    if submit_button:
        if MODEL is None: # Should have been caught earlier, but good check
            st.error("CRITICAL: Model is not available. Cannot predict.")
            return
        if SCALER is None and DF_EXPECTED_COLUMNS: # Only warn if scaling was expected
             st.warning("WARNING: Scaler is not available. Predictions will be attempted on unscaled data, which might be inaccurate if the model expects scaled inputs.")
        if not DF_EXPECTED_COLUMNS:
            st.error("CRITICAL: Expected feature column list is not defined. Cannot proceed with prediction.")
            return

        # Map string inputs to numerical
        gender = GENDER_MAPPING[gender_str]
        marital_status = MARITAL_STATUS_MAPPING[marital_status_str]
        application_mode = APPLICATION_MODE_MAPPING[application_mode_str]

        # Create DataFrame for single prediction
        data_dict = {
            'Marital_status': marital_status, 'Application_mode': application_mode,
            'Previous_qualification_grade': prev_qualification_grade,
            'Admission_grade': admission_grade, 'Displaced': displaced, 'Debtor': debtor,
            'Tuition_fees_up_to_date': tuition_fees_up_to_date, 'Gender': gender,
            'Scholarship_holder': scholarship_holder, 'Age_at_enrollment': age,
            'Curricular_units_1st_sem_enrolled': curricular_units_1st_sem_enrolled,
            'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved,
            'Curricular_units_1st_sem_grade': curricular_units_1st_sem_grade,
            'Curricular_units_2nd_sem_enrolled': curricular_units_2nd_sem_enrolled,
            'Curricular_units_2nd_sem_evaluations': curricular_units_2nd_sem_evaluations,
            'Curricular_units_2nd_sem_approved': curricular_units_2nd_sem_approved,
            'Curricular_units_2nd_sem_grade': curricular_units_2nd_sem_grade,
            'Curricular_units_2nd_sem_without_evaluations': curricular_units_2nd_sem_without_evaluations
        }
        input_df = pd.DataFrame([data_dict])
        
        with st.spinner("🤖 Analyzing data..."):
            processed_df = preprocess_data(input_df)
            if processed_df is not None:
                prediction_result = predict_status(processed_df)
                if prediction_result:
                    display_prediction_result(prediction_result[0])
                else:
                    st.error("Prediction failed after preprocessing.")
            else:
                st.error("Data preprocessing failed. Prediction cannot proceed.")

def multiple_prediction_tab():
    st.header("📊 Batch Prediction via File Upload")
    
    with st.expander("📜 User Guide & Template", expanded=False):
        st.markdown(f"""
            1.  **Download the Template:** Click the button below to get the Excel template (`{TEMPLATE_EXCEL_PATH.name}`).
            2.  **Fill Data:** Populate the template with student data. Ensure column names match the template.
                *   `ID` and `Name` columns are for your reference and will be included in the output. They are not used for prediction.
                *   Categorical fields (Gender, Marital Status, Application Mode) should use the exact string values shown as options in the 'Single Student Prediction' tab dropdowns.
            3.  **Upload File:** Upload the completed Excel file (.xlsx or .xls).
            4.  **Predict:** Click the "🚀 Predict from File" button.
            5.  **View & Download:** Results will be displayed and available for download.
        """)
        try:
            with open(TEMPLATE_EXCEL_PATH, "rb") as fp:
                st.download_button(
                    label="📥 Download Excel Template",
                    data=fp,
                    file_name=TEMPLATE_EXCEL_PATH.name,
                    mime="application/vnd.ms-excel",
                    help="Download student data Excel template"
                )
        except FileNotFoundError:
            st.warning(f"Template file '{TEMPLATE_EXCEL_PATH}' not found. Download button disabled.")

    uploaded_file = st.file_uploader(
        "📤 Upload Student Data File",
        type=['xlsx', 'xls'],
        help="Upload an Excel file with student data based on the template."
    )

    if uploaded_file:
        try:
            input_df_raw = pd.read_excel(uploaded_file)
            st.success(f"✅ File '{uploaded_file.name}' uploaded successfully with {len(input_df_raw)} rows!")

            # Keep ID and Name for later, ensure they exist or create placeholders
            if 'ID' in input_df_raw.columns:
                ids = input_df_raw['ID'].astype(str)
            else:
                st.info("No 'ID' column found in uploaded file. Generating sequential IDs.")
                ids = pd.Series([f"Row_{i+1}" for i in range(len(input_df_raw))], name="ID")
            
            if 'Name' in input_df_raw.columns:
                names = input_df_raw['Name']
            else:
                st.info("No 'Name' column found. Using 'N/A' for names.")
                names = pd.Series(["N/A"] * len(input_df_raw), name="Name")
            
            df_to_predict = input_df_raw.copy()
            
            # Rename columns from template style to model style (internal names)
            rename_map = {
                'Marital Status': 'Marital_status', 'Application Mode': 'Application_mode',
                'Previous Qualification Grade': 'Previous_qualification_grade',
                'Admission Grade': 'Admission_grade', 'Tuition up to date': 'Tuition_fees_up_to_date',
                'Scholarship': 'Scholarship_holder', 'Age at Enrollment': 'Age_at_enrollment',
                'Units 1st Semester Enrolled': 'Curricular_units_1st_sem_enrolled',
                'Units 1st Semester Approved': 'Curricular_units_1st_sem_approved',
                'Units 1st Semester Grade': 'Curricular_units_1st_sem_grade',
                'Units 2nd Semester Enrolled': 'Curricular_units_2nd_sem_enrolled',
                'Units 2nd Semester Approved': 'Curricular_units_2nd_sem_approved',
                'Units 2nd Semester Grade': 'Curricular_units_2nd_sem_grade',
                'Units 2nd Semester Evaluations': 'Curricular_units_2nd_sem_evaluations',
                'Units 2nd Semester No Evaluations': 'Curricular_units_2nd_sem_without_evaluations'
            }
            df_to_predict.rename(columns=rename_map, inplace=True) # Rename template columns to internal feature names

            # Map categorical features
            # Handle potential errors if mapping keys are not found (e.g., bad data in Excel)
            try:
                df_to_predict['Gender'] = df_to_predict['Gender'].map(GENDER_MAPPING)
                df_to_predict['Marital_status'] = df_to_predict['Marital_status'].map(MARITAL_STATUS_MAPPING)
                df_to_predict['Application_mode'] = df_to_predict['Application_mode'].map(APPLICATION_MODE_MAPPING)
            except KeyError as e:
                st.error(f"ERROR: Unrecognized value in uploaded file for a categorical column (e.g., Gender, Marital_status, Application_mode). Please check your data. Details: {e}")
                return # Stop processing

            # Check for NaNs after mapping, which indicate unmapped values
            if df_to_predict[['Gender', 'Marital_status', 'Application_mode']].isnull().any().any():
                st.error("ERROR: Some categorical values in the uploaded file could not be mapped (resulted in NaN). "
                         "Please ensure values in 'Gender', 'Marital Status', 'Application Mode' columns "
                         "match the allowed options exactly (case-sensitive).")
                return

            st.markdown("---")
            st.subheader("📂 Uploaded Data Preview (Raw)")
            preview_rows = st.slider('Select number of rows to preview (raw data):', 1, min(20, len(input_df_raw)), 5, key="batch_preview_slider")
            st.dataframe(input_df_raw.head(preview_rows))

            if st.button("🚀 Predict from File", use_container_width=True):
                if MODEL is None:
                    st.error("CRITICAL: Model is not available. Cannot predict.")
                    return
                if not DF_EXPECTED_COLUMNS:
                    st.error("CRITICAL: Expected feature column list is not defined. Cannot predict.")
                    return

                with st.spinner("🧠 Processing batch and making predictions..."):
                    # df_to_predict now has renamed and mapped columns.
                    # preprocess_data will select/order according to DF_EXPECTED_COLUMNS and scale.
                    processed_df_batch = preprocess_data(df_to_predict)
                    if processed_df_batch is not None:
                        predictions = predict_status(processed_df_batch)
                        if predictions:
                            result_df = pd.DataFrame({'ID': ids, 'Name': names, 'Predicted_Status': predictions})
                            st.subheader("📈 Prediction Results")
                            st.dataframe(result_df.style.applymap(get_color_for_status, subset=['Predicted_Status']))
                            
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                                result_df.to_excel(writer, sheet_name='Predictions', index=False)
                            
                            st.download_button(
                                label="💾 Download Prediction Results (Excel)",
                                data=excel_buffer.getvalue(),
                                file_name="student_predictions_output.xlsx",
                                mime="application/vnd.ms-excel",
                                help="Download the prediction results as an Excel file."
                            )
                            st.toast("Predictions complete and ready for download!", icon="✅")
                        else:
                             st.error("Batch prediction failed after preprocessing.")
                    else:
                        st.error("Batch data preprocessing failed.")
        
        except KeyError as e:
             st.error(f"💥 Error: A required column is missing from the uploaded Excel file or template mapping. Missing column: {e}. Please check your file against the template.")
        except Exception as e:
            st.error(f"💥 An error occurred while processing the uploaded file: {e}")
            st.error("Please ensure the file is a valid Excel file and matches the template structure. "
                     "Also check that all feature columns can be converted to numbers after mapping.")

# --- Main Application ---
def main():
    st.title(APP_TITLE)
    st.markdown("---")
    st.markdown(
        f"""
        Welcome to the Student Outcome Predictor for Jaya Jaya Institute! 
        This tool helps predict whether a student is likely to **Graduate** or **Dropout**.
        
        """
    )

    if MODEL is None: # Primary check after attempting to load
        st.error("🔴 CRITICAL ERROR: Model failed to load. The application cannot make predictions. Please check the `model_rf.joblib` file.")
        st.stop() # Stop execution if core components are missing
    
    if SCALER is None and DF_EXPECTED_COLUMNS : # Only warn if we expected a scaler but didn't get one
        st.warning(
            f"🟡 WARNING: Scaler could not be fitted (e.g., `{BASE_DATA_FOR_SCALING_PATH.name}` "
            "missing or has non-numeric feature columns). Predictions might be inaccurate if the model expects scaled data."
        )
    
    if not DF_EXPECTED_COLUMNS : # If this is None or empty after load attempt
         st.warning(
             f"🟡 WARNING: The list of expected feature columns could not be determined from `{BASE_DATA_FOR_SCALING_PATH.name}`. "
             "Using a default list, which might cause errors or inaccuracies if it doesn't match your model."
        )

    tab_single, tab_multiple = st.tabs(["👤 Single Student Prediction", "👥 Batch Prediction (File Upload)"])

    with tab_single:
        single_prediction_tab()

    with tab_multiple:
        multiple_prediction_tab()
    
    st.markdown("---")
    current_year = datetime.date.today().year
    # Adjust start year if your project started earlier
    copyright_year = "2025" if current_year <= 2025 else f"2024 - {current_year}" 
    st.caption(f"© {copyright_year} Jaya Jaya Institute | Developed with Streamlit by Frans Gabriel Sianturi. ([LinkedIn](http://linkedin.com/in/fransgs))")

if __name__ == '__main__':
    main()
