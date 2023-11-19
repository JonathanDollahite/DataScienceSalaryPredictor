import pandas as pd
import plotly_express as px
from sklearn.linear_model import Ridge 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from dash import Dash, dcc, html, Input, Output, State

# Read in the data, drop redundant variables
salaries_df = pd.read_csv('ds_salaries.csv')
salaries_df.drop(['salary','salary_currency'], axis = 1, inplace = True)

# Drop the target variable and set is as the label
X = salaries_df.drop('salary_in_usd', axis=1)  
y = salaries_df['salary_in_usd']

# Create a column transformer
categorical_columns = ['work_year', 'experience_level', 'employment_type', 'job_title', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']  

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns)
    ],
    remainder='passthrough'
)

# Apply one-hot encoding
X_encoded = preprocessor.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.20, random_state=42)

# Create an instance of the ridge regression model and fit it to the training data
ridge = Ridge(alpha=1.0) 
ridge.fit(X_train, y_train)

# Set the style for the page
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Create the app
app = Dash(__name__, external_stylesheets=external_stylesheets)\

# Include this line for use with render.com
server = app.server

# Include app title
app.title = "Data Science Salary Predictor"

# Get unique values for each column
unique_values = {column: salaries_df[column].unique().tolist() for column in salaries_df.columns}

# Create the app layout
app.layout = html.Div([
    html.H1('Data Science Salary Predictor'),
    html.Div([
        html.Div([
            html.Label('Work Year', style={'display': 'inline-block', 'width': '200px'}),
            dcc.Dropdown(
                id='work-year',
                options=[{'label': value, 'value': value} for value in unique_values['work_year']],
                value=unique_values['work_year'][0]
            ),
        ]),
        html.Div([
            html.Label('Experience Level', style={'display': 'inline-block', 'width': '200px'}),
            dcc.Dropdown(
                id='experience-level',
                options=[{'label': value, 'value': value} for value in unique_values['experience_level']],
                value=unique_values['experience_level'][0]
            ),
        ]),
        html.Div([
            html.Label('Employment Type', style={'display': 'inline-block', 'width': '200px'}),
            dcc.Dropdown(
                id='employment-type',
                options=[{'label': value, 'value': value} for value in unique_values['employment_type']],
                value=unique_values['employment_type'][0]
            ),
        ]),
        html.Div([
            html.Label('Job Title', style={'display': 'inline-block', 'width': '200px'}),
            dcc.Dropdown(
                id='job-title',
                options=[{'label': value, 'value': value} for value in unique_values['job_title']],
                value=unique_values['job_title'][0]
            ),
        ]),
        html.Div([
            html.Label('Employee Residence', style={'display': 'inline-block', 'width': '200px'}),
            dcc.Dropdown(
                id='employee-residence',
                options=[{'label': value, 'value': value} for value in unique_values['employee_residence']],
                value=unique_values['employee_residence'][0]
            ),
        ]),
        html.Div([
            html.Label('Remote Ratio', style={'display': 'inline-block', 'width': '200px'}),
            dcc.Dropdown(
                id='remote-ratio',
                options=[{'label': value, 'value': value} for value in unique_values['remote_ratio']],
                value=unique_values['remote_ratio'][0]
            ),
        ]),
        html.Div([
            html.Label('Company Location', style={'display': 'inline-block', 'width': '200px'}),
            dcc.Dropdown(
                id='company-location',
                options=[{'label': value, 'value': value} for value in unique_values['company_location']],
                value=unique_values['company_location'][0]
            ),
        ]),
        html.Div([
            html.Label('Company Size', style={'display': 'inline-block', 'width': '200px'}),
            dcc.Dropdown(
                id='company-size',
                options=[{'label': value, 'value': value} for value in unique_values['company_size']],
                value=unique_values['company_size'][0]
            ),
        ]),
    ]),  

    html.Div([
        html.Button('Predict', id='predict-button', n_clicks=0),
    ]),

    html.Label('Predicted Salary', style={'font-weight': 'bold'}),
    html.Div(id='predicted-salary')
])


# Create the callback
@app.callback(
    Output('predicted-salary', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('work-year', 'value'),
    State('experience-level', 'value'),
    State('employment-type', 'value'),
    State('job-title', 'value'),
    State('employee-residence', 'value'),
    State('remote-ratio', 'value'),
    State('company-location', 'value'),
    State('company-size', 'value'),]    
)
def predict_salary(n_clicks, work_year, experience_level, employment_type, job_title, employee_residence, remote_ratio, company_location, company_size):
    
    # Create input features DataFrame with column names
    input_features_df = pd.DataFrame({
        'work_year': [work_year],
        'experience_level': [experience_level],
        'employment_type': [employment_type],
        'job_title': [job_title],
        'employee_residence': [employee_residence],
        'remote_ratio': [remote_ratio],
        'company_location': [company_location],
        'company_size': [company_size]
    })

    # Transform input features using the preprocessor
    input_features_transformed = preprocessor.transform(input_features_df)

    # Predict the salary
    predicted_salary = ridge.predict(input_features_transformed)[0]
    
    return html.Div(
        children=[
            html.Div(f"${predicted_salary:,.2f}", style={'border': '2px solid black', 'padding': '10px', 'width': '48%', 'float': 'left'})
        ]
    )

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


