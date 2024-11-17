import os


class Config:
    SQLALCHEMY_DATABASE_URI = 'postgresql://pompos02:mypassword123@localhost:5432/speeches'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
