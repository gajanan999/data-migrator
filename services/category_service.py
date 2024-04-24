import os
import joblib


class CategoryService:

    def __init__(self, user_category_selection_repository):
        self.user_category_selection_repository = user_category_selection_repository

    def getCategory(self, user_id, feature1, feature2, feature3):

        # Check if user has category selection in the database
        category = self.user_category_selection_repository.get_user_selected_transactions_category(user_id, feature1,
                                                                                                   feature2, feature3)

        if category != '':
            return category
        else:
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

            input_data = feature1 + ' ' + feature2 + ' ' + feature3
            # If the input data is a single string, convert it to a list of one element
            if isinstance(input_data, str):
                input_data = [input_data]
            prediction = loaded_model.predict(input_data).tolist()
            print('Prediction', prediction)
            if len(prediction) > 0:
                return prediction[0]
            else:
                return 'Other'
