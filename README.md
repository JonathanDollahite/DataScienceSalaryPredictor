# Data Science Salary Predictor

## Overview
This interactive web application, powered by Dash, predicts salaries in the field of data science based on various input features. The predictive model utilizes Ridge Regression and is trained on a dataset containing information about data science professionals' salaries.

You can visit the functional app at the following link: https://data-science-salary-predictor.onrender.com/

More information about the dataset can be found at the following link: https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023

## Getting Started
Clone the repository, navigate into the folder, and ensure you have the necessary libraries installed. Use the following command to do all three at once (assuming HTTPS):

```bash
git clone https://github.com/JonathanDollahite/DataScienceSalaryPredictor.git;\
cd DataScienceSalaryPredictor;\
pip install -r requirements.txt
```

After installing the dependencies, run the app with the following command:

```bash
python app.py
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
