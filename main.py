import json
import os
import csv
import random

from flask import Flask, request, render_template, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from db.user_repository import UserRepository
from db.user_category_selection_repository import UserCategorySelectionRepository
from db.transaction_training_data_repository import TransactionTrainingDataRepository
from services.category_service import CategoryService
from services.training_service import TrainingService
app = Flask(__name__)
scheduler = BackgroundScheduler()

db_uri = 'sqlite:///data_migrator.db'
user_repository = UserRepository(db_uri)
user_category_selection_repository = UserCategorySelectionRepository(db_uri)
transaction_training_data_repository = TransactionTrainingDataRepository(db_uri)
category_service = CategoryService(user_category_selection_repository)
training_service = TrainingService()

@app.route("/users")
def getUsers():
    user_repository.add_user('John',31)
    return user_repository.get_all_users()

@app.route("/")
def hello():
    return "Hello User, This is Data Migrator Project "

def append_user_selection_to_training_data_table():
    try:
        user_category_selections = user_category_selection_repository.get_all_user_selected_transactions_categories()
        print(user_category_selections)
        # Convert the query result into a DataFrame
        df = pd.DataFrame(user_category_selections)
        training_service.train_model(df)
        # print('Dataframe:\n', df.to_string())
        # if transaction_training_data_repository.delete_all_records():
        #     entities = training_service.create_entities_from_dataframe(df)
        #     if transaction_training_data_repository.add_transaction_training_data(entities):
        #         print('training data collected from the table UserSelectedTransactionsCategories')
        #         train_model_3()
        #     else:
        #         raise Exception('Training Data collection failed')

    except Exception as e:
        print(e)

def train_model_3():
    print('Model Training Started')

def append_csv_data_to_excel():
    # Define the folder paths
    input_folder = 'user-submitted-category-files'

    output_folder = 'trainings_data'
    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the output filename with date-time tag
    output_filename = f"training_data_{current_datetime}.csv"

    # Get a list of all files in the input folder
    input_files = os.listdir(input_folder)

    # Initialize an empty list to store all DataFrames
    all_dataframes = []

    # Loop through each file in the input folder
    for filename in input_files:
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            try:
                # Read the data from the current file with 'Latin-1' encoding
                df = pd.read_csv(file_path, encoding='Latin-1')
            except UnicodeDecodeError:
                # If 'Latin-1' fails, try reading with 'utf-8' and ignore any errors
                df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')

            # Append the DataFrame to the list
            all_dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    all_data = pd.concat(all_dataframes, ignore_index=True)

    # Save the DataFrame to an Excel file
    output_filepath = os.path.join(output_folder, output_filename)
    all_data.to_csv(output_filepath, index=False)

    print(f"Data appended and saved to '{output_filename}'")


@app.route('/start-scheduler')
def start_scheduler():
    # Schedule the job to run every 15 minutes
    if not scheduler.running:
        scheduler.start()
        scheduler.add_job(id='append_csv_data', func=append_user_selection_to_training_data_table, trigger='interval', seconds=15)
        #scheduler.start()
        return "Scheduler started.  "
    else:
        return "Schedular already started"


@app.route('/stop-scheduler')
def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown()
        return "Scheduler stopped."
    else:
        return "Scheduler is not running"


@app.route('/train-model', methods=['GET'])
def train_model():
    try:
        # Define the folder paths
        input_folder = 'trainings_data'
        output_folder = 'trained_data'
        output_folder_model = 'models'

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder_model):
            os.makedirs(output_folder_model)

        # Get a list of all files in the input folder
        input_files = os.listdir(input_folder)


        # Initialize an empty list to store all DataFrames
        all_dataframes = []

        # Loop through each file in the input folder
        for filename in input_files:
            if filename.endswith('.csv'):
                file_path = os.path.join(input_folder, filename)

                df = pd.read_csv(file_path)

                # Append the DataFrame to the list
                all_dataframes.append(df)

                # Move the file to the output folder
                output_file_path = os.path.join(output_folder, filename)
                #shutil.copy(file_path, output_file_path)

        # Concatenate all DataFrames into a single DataFrame
        all_data = pd.concat(all_dataframes, ignore_index=True)

        # Replace NaN with '-' in the 'Beguenstigter', 'buchungstext', and 'verwendungszweck' columns
        all_data['Beguenstigter'].fillna('-', inplace=True)
        all_data['buchungstext'].fillna('-', inplace=True)
        all_data['verwendungszweck'].fillna('-', inplace=True)

        all_data['buchungstext'] = all_data['buchungstext'].str.replace('[^a-zA-Z0-9\s]', '')  # Remove special characters
        all_data['verwendungszweck'] = all_data['verwendungszweck'].str.replace('[^a-zA-Z0-9\s]', '')  # Remove special characters
        all_data['Beguenstigter'] = all_data['Beguenstigter'].str.replace('[^a-zA-Z0-9\s]', '')
        all_data['buchungstext'] = all_data['buchungstext'].str.lower()  # Lowercase the text
        all_data['verwendungszweck'] = all_data['verwendungszweck'].str.lower()  # Lowercase the text
        all_data['Beguenstigter'] = all_data['Beguenstigter'].str.lower()
        all_data['features'] = all_data['Beguenstigter'] + ' ' + all_data['buchungstext'] + ' ' + all_data['verwendungszweck']  # Concatenate buchungstext and verwendungszweck
        print(all_data['features'])
        # Split the data into training and testing sets
        X = all_data['features']
        y = all_data['category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', LinearSVC())
        ])
        model.fit(X_train, y_train)
        print('Model Trained')
        # Save the trained model for future predictions
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"trained_model_{current_datetime}.joblib"
        model_filepath = os.path.join(output_folder_model, model_filename)
        joblib.dump(model, model_filepath)
        print('Model dumped')
        # Return a success response
        response = {'message': 'Model trained and files moved successfully.', 'model_filepath': model_filepath}
        return jsonify(response), 200

    except Exception as e:
        # If any error occurs, return an error response
        print(e)
        error_response = {'error': str(e)}
        return jsonify(error_response), 500


@app.route('/train-model-demo', methods=['GET'])
def train_model_2():
    # Generating random data for the dataframe
    data = {
        'Height': [random.uniform(150, 200) for _ in range(100)],
        'Age': [random.randint(18, 80) for _ in range(100)],
        'Weight': [random.uniform(50, 100) for _ in range(100)]
    }

    # Creating the dataframe
    df = pd.DataFrame(data)
    print(df.head())

    # Splitting into features and target variable
    X = df[['Height', 'Age']]
    y = df['Weight']

    # Splitting the data into training and testing sets (80% for training, 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize the Linear Regression model
    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Compare the predicted values with the actual values
    results = pd.DataFrame({'Actual Weight': y_test, 'Predicted Weight': y_pred})
    print(results.head())
    return results.to_json()


@app.route("/data-migrator")
def upload_html():
    # Render the categories template with the generated categories
    return render_template('index.html')


@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    # Get the file from the request
    file = request.files['file']

    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract the file extension (assuming the file has an extension)
    _, file_extension = os.path.splitext(file.filename)

    # Generate a unique filename with date-time tag
    unique_filename = f"training_data_{current_datetime}{file_extension}"

    # Define the folder to save the file in
    folder_path = 'trainings_data'

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Combine the folder path and the unique filename
    file_path = os.path.join(folder_path, unique_filename)

    # Save the file with the new filename and path
    file.save(file_path)

    # Process the training data here (if needed)

    # Return a response
    return jsonify({'message': 'Training data uploaded successfully.'})


@app.route('/submit-categories', methods=['POST'])
def submit_categories():
    try:
        data = request.get_json()  # Get the JSON data from the request body
        #
        # # Process or store the received data as per your requirements
        # # For demonstration purposes, we'll just print the data and save it to a CSV file
        # print("Received data from the client:")
        # print(data)
        #
        # # Save the received data to a CSV file in the userSubmitted folder
        # folder_path = 'user-submitted-category-files'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        #
        # # Get the current date and time
        # current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        #
        # # Append the date-time tag to the file name
        # csv_file_name = f'submitted_categories_{current_datetime}.csv'
        #
        # csv_file_path = os.path.join(folder_path, csv_file_name)
        # with open(csv_file_path, mode='w', newline='') as csvfile:
        #     fieldnames = ['category', 'Beguenstigter', 'buchungstext', 'verwendungszweck', 'total_spent']
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #
        #     writer.writeheader()
        #     for row in data:
        #         writer.writerow(row)
        # # Return a success response to the client
        # response = {'message': 'Categories submitted successfully and CSV file created in userSubmitted folder.'}
        # return jsonify(response), 200

        for row in data:
            user_category_selection_repository.add_user_selected_transactions_category(1, row['recipient'], row['booking_text'], row['purpose_of_transaction'], row['category'])

        print('Data Added into DB')
        user_category_selection_repository.close_connection()

        print('categories:', user_category_selection_repository.get_all_user_selected_transactions_categories())
        print('categories length:{}',
              len(user_category_selection_repository.get_all_user_selected_transactions_categories()))
        return 'true'
    except Exception as e:
        # If any error occurs, return an error response to the client
        print(e)
        error_response = {'error': str(e)}
        return jsonify(error_response), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file.save('temp.xlsx')  # Save the file to a temporary location

    # Process the Excel file
    df = pd.read_excel('temp.xlsx')

    # Convert the columns to strings
    df['Buchungstext'] = df['Buchungstext'].astype(str)
    df['Verwendungszweck'] = df['Verwendungszweck'].astype(str)
    df['Beguenstigter/Zahlungspflichtiger'] = df['Beguenstigter/Zahlungspflichtiger'].astype(str)
    # Generate categories for transactions
    categories = {
        'Food and Drink': ['restaurant', 'groceries','TAJ MAHAL'],
        'Sport': ['gym', 'fitness','Novalnet AG'],
        'Home': ['rent', 'mortgage', 'utilities'],
        'Life': ['insurance', 'healthcare'],
        'Investment':['eToro'],
        'Transportation': ['car', 'fuel', 'public transport'],
        'Utilities': ['electricity', 'water', 'internet','Energy', 'Lebara Germany Limited'],
        'Self Expense': ['PRASAD'],
        'Other': ['Other']
    }

    main_categories = list(categories.keys())

    # Generate categories for transactions
    categories1 = df['Buchungstext'].unique().tolist()


    user_id=1
    # Categorize the transactions based on Buchungstext and Verwendungszweck
    #df['Category'] = df.apply(lambda row: categorize_transaction(row['Beguenstigter/Zahlungspflichtiger'], row['Verwendungszweck'], categories), axis=1)
    df['Category'] = df.apply(lambda row:  category_service.getCategory(user_id, row['Beguenstigter/Zahlungspflichtiger'], row['Buchungstext'], row['Verwendungszweck'] ), axis=1)

    # Convert the 'column_name' to strings and replace NaN with an empty string
    df['Beguenstigter/Zahlungspflichtiger'] = df['Beguenstigter/Zahlungspflichtiger'].fillna('').astype(str)
    # Convert the 'column_name' to strings and replace NaN with an empty string
    df['Buchungstext'] = df['Buchungstext'].fillna('').astype(str)
    df['Verwendungszweck'] = df['Verwendungszweck'].fillna('').astype(str)
    print(df.to_string())
    # Group the data by 'Buchungstext' and 'Verwendungszweck' and calculate the total spent
    grouped_data = df.groupby(['Category','Beguenstigter/Zahlungspflichtiger','Buchungstext', 'Verwendungszweck'])['Betrag'].sum()

    df['Betrag'] = df['Betrag'].astype(str)
    df['Betrag'] = df['Betrag'].apply(lambda x: x.replace('EUR', '')).apply(lambda x: x.replace('.', '')).apply(lambda x: x.replace(',', '.')).astype('float')

    # Group the data by the Category column and calculate the sum of the Betrag column for each category
    category_grouped_data = df.groupby('Category')['Betrag'].sum().reset_index()

    return render_template('categories.html', main_categories=main_categories, categories=categories1, grouped_data=grouped_data, category_grouped_data=category_grouped_data)


def categorize_transaction(Beguenstigter, verwendungszweck, categories):
    for category, keywords in categories.items():
        print(keywords)
        for keyword in keywords:
            if keyword.lower() in Beguenstigter.lower() or keyword.lower() in verwendungszweck.lower():
                return category
    return 'Other'



@app.route('/predict', methods=['GET'])
def predict():
    # Define the models directory path
    models_directory = 'models'

    # Get the list of all files in the models directory
    model_files = os.listdir(models_directory)

    print(model_files)
    # Sort the model files based on their creation time (modification time)
    latest_model_file = max(model_files, key=lambda x: os.path.getmtime(os.path.join(models_directory, x)))

    print('Latest file', latest_model_file)
    # Load the latest model
    latest_model_path = os.path.join(models_directory, latest_model_file)
    loaded_model = joblib.load(latest_model_path)

    input_data = 'ONLINE-UEBERWEISUNG' + ' ' + 'PVA Wertlau DATUM 27.06.2019, 22.18 UHR1.TAN 702450'
    # If the input data is a single string, convert it to a list of one element
    if isinstance(input_data, str):
        input_data = [input_data]
    prediction = loaded_model.predict(input_data).tolist()
    print('Prediction', prediction)

    # Return the prediction result as JSON
    return json.dumps(prediction)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


