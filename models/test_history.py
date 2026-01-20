from ..extensions import db
from datetime import datetime
import json

class TestHistory(db.Model):
    __tablename__ = 'test_history'
    id = db.Column(db.Integer, primary_key=True)
    test_name = db.Column(db.String(128), nullable=False)
    parameters = db.Column(db.Text, nullable=True)  # json
    result = db.Column(db.Text, nullable=True)      # json
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
