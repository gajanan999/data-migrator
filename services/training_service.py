import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from db.database_handler import TransactionTrainingData


class TrainingService:

    def __init__(self):
        self.input_folder = 'trainings_data'
        self.output_folder = 'trained_data'
        self.output_folder_model = 'models'

        print('Training Service Initialized')

    def create_entities_from_dataframe(self, df):
        entities = []
        for index, row in df.iterrows():
            entity = TransactionTrainingData(
                recipient=row['recipient'],
                booking_text=row['booking_text'],
                purpose_of_transaction=row['purpose_of_transaction'],
                category=row['category']
            )
            entities.append(entity)
        return entities

    def create_input_and_output_folder(self):
        # Create the output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Create the output folder if it doesn't exist
        if not os.path.exists(self.output_folder_model):
            os.makedirs(self.output_folder_model)

    def train_model(self, df):
        print('')
        try:
            self.create_input_and_output_folder()

            df['recipient'].fillna('-', inplace=True)
            df['booking_text'].fillna('-', inplace=True)
            df['purpose_of_transaction'].fillna('-', inplace=True)

            df['recipient'] = df['recipient'].str.replace('[^a-zA-Z0-9\s]', '')
            df['booking_text'] = df['booking_text'].str.replace('[^a-zA-Z0-9\s]', '')  # Remove special characters
            df['purpose_of_transaction'] = df['purpose_of_transaction'].str.replace('[^a-zA-Z0-9\s]',
                                                                                    '')  # Remove special characters

            df['recipient'] = df['recipient'].str.lower()
            df['booking_text'] = df['booking_text'].str.lower()  # Lowercase the text
            df['purpose_of_transaction'] = df['purpose_of_transaction'].str.lower()  # Lowercase the text

            df['features'] = df['recipient'] + ' ' + df['booking_text'] + ' ' + df[
                'purpose_of_transaction']  # Concatenate buchungstext and verwendungszweck
            print('features: \n', df['features'])

            # Split the data into training and testing sets
            X = df['features']
            y = df['category']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            model = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('classifier', LinearSVC())
            ])
            model.fit(X_train, y_train)
            print('LinearSVC Model Trained')
            # Save the trained model for future predictions
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"trained_model_{current_datetime}.joblib"
            model_filepath = os.path.join(self.output_folder_model, model_filename)
            joblib.dump(model, model_filepath)
            print('Model dumped at {}', model_filepath)
        except Exception as e:
            print(e)
