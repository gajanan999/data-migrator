
# my_repository_package/user_repository.py
from .database_handler import DatabaseHandler
from flask import jsonify

class UserRepository:

    def __init__(self, db_uri):
        self.db_handler = DatabaseHandler(db_uri)

    def add_user(self, name, age):
        self.db_handler.add_user(name, age)

    def get_all_users(self):
        users = self.db_handler.get_all_users()
        for user in users:
            print(f'User ID: {user.id}, Name: {user.name}, Age: {user.age}')

        # Update a user's age
        # database_handler.update_user_age(name='John Doe', new_age=31)

        users_data = []
        for user in users:
            user_data = {
                'id': user.id,
                'name': user.name,
                'age': user.age
            }
            users_data.append(user_data)
        return jsonify(users_data)

    def update_user_age(self, name, new_age):
        return self.db_handler.update_user_age(name, new_age)

    def close_connection(self):
        self.db_handler.close_connection()
