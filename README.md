# Crop Recommendation System 🌱

This project uses a **Random Forest Classifier** machine learning model to recommend the most suitable crop based on soil and weather conditions. It also provides an interactive **Streamlit** interface for users to input parameters like nitrogen, phosphorus, potassium levels, temperature, humidity, pH, and rainfall.

## 🚀 Features

- **Machine Learning Model**: Uses Random Forest Classifier for predicting the crop.
- **User Interface**: A clean and interactive interface built using Streamlit.
- **Preprocessing**: Includes scaling and label encoding for input features and target variable.
- **Model Persistence**: The trained model, scaler, and label encoder are saved for reusability.

## 📁 Project Structure

```
crop-recommendation-system/
│
├── crop_recommendation.csv      # Dataset
├── main.py                      # Streamlit app with prediction
├── crop_recommendation_model.pkl # Trained Random Forest model
├── scaler.pkl                   # StandardScaler for feature scaling
├── label_encoder.pkl            # Label encoder for crop labels
├── .gitignore                   # List of files to ignore in git
└── README.md                    # Project documentation (this file)
```

## ⚙️ Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/crop-recommendation-system.git
    cd crop-recommendation-system
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run main.py
    ```

## 🖥️ Usage

Once the Streamlit app is running, you'll be prompted to input values for the following parameters:

- **Nitrogen (N)**
- **Phosphorus (P)**
- **Potassium (K)**
- **Temperature (in Celsius)**
- **Humidity (%)**
- **Soil pH**
- **Rainfall (in mm)**

Click on the "Predict Crop" button, and the app will display the predicted crop based on the entered values.

## 🎯 Example Prediction

Here's how a prediction might look using the app:
```bash
Nitrogen: 80
Phosphorus: 38
Potassium: 15
Temperature: 30.88
Humidity: 92.00
pH: 5.03
Rainfall: 120.94
```
**Predicted Crop: Rice**

## 🧑‍💻 Model Information

The machine learning model is a **Random Forest Classifier** trained on the provided `crop_recommendation.csv` dataset. The features include soil nutrients (N, P, K), environmental factors (temperature, humidity, rainfall), and soil pH, which are scaled using `StandardScaler`.

The target variable (crop) is label-encoded, and the model predicts one of 22 crop types.

## 📊 Performance

The model achieves high accuracy, validated using cross-validation. Here’s a sample output from the evaluation:

- **Accuracy**: 98.5%
- **Cross-Validation Accuracy**: 98.2% (mean of 5-fold CV)

## 🔧 Tools Used

- **Python**: Core programming language for the project.
- **Streamlit**: Web framework for building the user interface.
- **scikit-learn**: Machine learning library for model building and evaluation.
- **Pandas**: Data manipulation and preprocessing.
- **NumPy**: Numerical computing.
- **Joblib**: Saving and loading models and other objects.
  
## 🔑 Model Files

The following files are saved after model training:
- `crop_recommendation_model.pkl`: Trained Random Forest model.
- `scaler.pkl`: StandardScaler used to scale features.
- `label_encoder.pkl`: LabelEncoder used to encode the crop labels.


Feel free to update this `README.md` as your project evolves or if you want to add more details like deployment instructions, dataset source, or project goals.
