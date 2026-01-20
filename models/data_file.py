from ..extensions import db
from datetime import datetime

class DataFile(db.Model):
    __tablename__ = 'data_files'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.Column(db.String(128), nullable=True)
    metadata = db.Column(db.Text, nullable=True)
