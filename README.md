# Data Science Salary Predictor

## Overview
This interactive web application, powered by Dash, predicts salaries in the field of data science based on various input features. The predictive model utilizes Ridge Regression and is trained on a dataset containing information about data science professionals' salaries.

## Getting Started
To run this app locally, make sure you have Python and the required libraries installed. Use the following commands to install the necessary dependencies:

```bash
pip install pandas plotly-express scikit-learn dash
```

After installing the dependencies, run the app with the following command:

```bash
python your_app_name.py
```

Visit [http://localhost:8050/](http://localhost:8050/) in your web browser to access the app.

## Usage
- **Input Features:** Select various parameters such as work year, experience level, employment type, job title, employee residence, remote ratio, company location, and company size.
- **Predict:** Click the "Predict" button to calculate the estimated salary based on the provided inputs.
- **Results:** View the predicted salary displayed on the page.

## App Structure
- The app layout includes dropdowns for each input feature and a "Predict" button.
- The predicted salary is dynamically updated upon button click using a Ridge Regression model.
- Styles are enhanced using external CSS for a clean and user-friendly interface.

## Acknowledgments
- This app utilizes the Dash framework for Python. Visit [Dash Documentation](https://dash.plotly.com/) for more information.
- The predictive model employs Ridge Regression from scikit-learn.
- Feel free to explore and modify the code to suit your specific needs. Happy predicting!
