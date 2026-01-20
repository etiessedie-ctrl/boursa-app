from ..extensions import db
from datetime import datetime

class PredictHistory(db.Model):
    __tablename__ = 'predict_history'
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(128), nullable=False)
    input_summary = db.Column(db.Text, nullable=True)
    output_summary = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
