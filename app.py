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
ISO_codes_df = pd.read_csv('ISO_codes.csv')

# Create a mapping from ISO codes to country names
iso_to_country = ISO_codes_df.set_index('alpha-2')['country'].to_dict()

# Replace ISO codes with country names in 'employee_residence' and 'company_location' columns
salaries_df['employee_residence'] = salaries_df['employee_residence'].map(iso_to_country)
salaries_df['company_location'] = salaries_df['company_location'].map(iso_to_country)

# Replace abbreviations with meaningful names
def replace_values(df, column, replacements):
    df[column] = df[column].replace(replacements)

# Define the replacements
experience_level_replacements = {
    'EN': 'Entry-level/Junior',
    'MI': 'Mid-level/Intermediate',
    'SE': 'Senior-level/Expert',
    'EX': 'Executive-level/Director'
}

employment_type_replacements = {
    'FT': 'Full-time',
    'CT': 'Contract',
    'FL': 'Freelance',
    'PT': 'Part-time'
}

company_size_replacements = {
    'S': 'Small',
    'M': 'Medium',
    'L': 'Large'
}

# Apply the replacements
replace_values(salaries_df, 'experience_level', experience_level_replacements)
replace_values(salaries_df, 'employment_type', employment_type_replacements)
replace_values(salaries_df, 'company_size', company_size_replacements)

# Drop the target variable and set is as the label
X = salaries_df.drop('salary_in_usd', axis=1)  
y = salaries_df['salary_in_usd']

# Create a column transformer
categorical_columns = [
    'work_year', 
    'experience_level', 
    'employment_type', 
    'job_title', 
    'employee_residence', 
    'remote_ratio', 
    'company_location', 
    'company_size'
]  

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
app = Dash(__name__, external_stylesheets=external_stylesheets)

# Include this line for use with render.com
server = app.server

# Include app title
app.title = "Data Science Salary Predictor"
app.description = "Predict your salary as a data scientist"

# Get unique values for each column
unique_values = {column: salaries_df[column].unique().tolist() for column in salaries_df.columns}

# Map column names to more human-friendly labels
column_name_mapping = {col: col.replace('_', ' ').title() for col in salaries_df.columns}
column_name_mapping['salary_in_usd'] = 'Salary in USD'

def create_predictor_selection(id, label, options, default_value):
    return html.Div([
        html.Label(label, style={'display': 'inline-block', 'width': '200px'}),
        dcc.Dropdown(
            id=id,
            options=[{'label': value, 'value': value} for value in options],
            value=default_value
        ),
    ])

predictor_selections = [
    create_predictor_selection('work-year', 'Work Year', unique_values['work_year'], 2023),
    create_predictor_selection('experience-level', 'Experience Level', unique_values['experience_level'], 'Entry-level/Junior'),
    create_predictor_selection('employment-type', 'Employment Type', unique_values['employment_type'], 'Full-time'),
    create_predictor_selection('job-title', 'Job Title', unique_values['job_title'], 'Data Scientist'),
    create_predictor_selection('employee-residence', 'Employee Residence', unique_values['employee_residence'], 'United States'),
    create_predictor_selection('remote-ratio', 'Remote Ratio', unique_values['remote_ratio'], 0),
    create_predictor_selection('company-location', 'Company Location', unique_values['company_location'], 'United States'),
    create_predictor_selection('company-size', 'Company Size', unique_values['company_size'], 'Medium'),
]

# Create the app layout
app.layout = html.Div(
    children=[
        html.H1("Data Science Salary Predictor", style={'text-align': 'center'}),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.H2("Predict Your Salary", style={'text-align': 'center'}),
                        html.Div(predictor_selections, className='row'),
                        html.Button('Predict Salary', id='predict-button', style={'background-color':'#0000A0', 'color':'white', 'font-weight': 'bold', 'justify-content': 'center', 'align-items': 'center'}),
                        html.Label('Predicted Salary', style={'font-weight': 'bold'}),
                        html.Div(id='predicted-salary')  
                    ],
                    style={'width': '20%', 'padding-right': '10px'}
                ),
                html.Div(
                    children=[
                        html.H2(id='comparison-title', style={'text-align': 'center'}),
                        html.Label("Choose a predictor variable", style={'font-weight': 'bold', 'text-align': 'center'}),
                        dcc.Dropdown(
                            id='predictor-variable-dropdown',
                            options=[{'label': column_name_mapping[col], 'value': col} for col in salaries_df.columns if col != 'salary_in_usd'],
                            value=X.columns[0],
                            style={'text-align': 'center'}
                        ),
                        dcc.Graph(id='scatterplot')
                    ],
                    style={'width': '80%', 'padding-left': '10px'}
                )
            ],
            style={'display': 'flex'}
        )
    ],
    style={'margin-top': '75px'}
)

# Create the callback for the predicted salary
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
def predict_salary(
    n_clicks, 
    work_year, 
    experience_level, 
    employment_type, 
    job_title, 
    employee_residence, 
    remote_ratio, 
    company_location,
    company_size
):
    
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
    
    return [html.Div(f"${predicted_salary:,.2f}", style={
                                                    'border': '2px solid black', 
                                                    'padding': '10px', 
                                                    'width': '48%', 
                                                    'float': 'left'
                                                    }
                )
    ]

# Define a callback function that will update the graph
@app.callback(
    Output('scatterplot', 'figure'),
    Input('predictor-variable-dropdown', 'value')
)
def update_graph(selected_predictor):
    ordered_categories = None
    if selected_predictor in ['employment_type', 'job_title', 'employee_residence', 'company_location']:
        ordered_categories = salaries_df.groupby(selected_predictor)['salary_in_usd'].median().sort_values().index.tolist()

    salaries_df[selected_predictor] = salaries_df[selected_predictor].astype(str)

    fig = px.box(
        salaries_df, x=selected_predictor, y='salary_in_usd',
        labels=column_name_mapping,
        category_orders={
            selected_predictor: ordered_categories,
            'work_year': ['2020', '2021', '2022', '2023'],
            'experience_level': ['Entry-level/Junior', 'Mid-level/Intermediate', 'Senior-level/Expert', 'Executive-level/Director'],
            'employment_type': ['Part-time', 'Freelance', 'Contract', 'Full-time'],
            'company_size': ['Small', 'Medium', 'Large'],
            'remote_ratio': ['0', '50', '100']
        },
    )
    return fig

# Define a callback function that will update the heading title
@app.callback(
    Output('comparison-title', 'children'),
    Input('predictor-variable-dropdown', 'value')
)
def update_title(selected_predictor):
    return f"Salary Compared to {column_name_mapping[selected_predictor]}"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
