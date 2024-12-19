from app import db


class Speech(db.Model):
    __tablename__ = 'speeches'
    id = db.Column(db.Integer, primary_key=True)
    member_name = db.Column(db.String, nullable=False)
    sitting_date = db.Column(db.DateTime, nullable=False)
    parliamentary_period = db.Column(db.String, nullable=False)
    parliamentary_session = db.Column(db.String, nullable=False)
    parliamentary_sitting = db.Column(db.String, nullable=False)
    political_party = db.Column(db.String, nullable=False)
    government = db.Column(db.String, nullable=True)
    member_region = db.Column(db.String, nullable=False)
    roles = db.Column(db.String, nullable=True)
    member_gender = db.Column(db.String, nullable=False)
    speaker_info = db.Column(db.String, nullable=True)
    speech = db.Column(db.Text, nullable=False)
