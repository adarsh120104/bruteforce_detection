import numpy as np
from flask import Flask, render_template, request, jsonify, abort, redirect, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
from sklearn.ensemble import IsolationForest
import datetime
import redis

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///brute_force.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['REDIS_URL'] = 'redis://localhost:6379/0'
limiter = Limiter(app)
db = SQLAlchemy(app)
redis_store = redis.StrictRedis.from_url(app.config['REDIS_URL'])

# get whitelsted IPs from a config file
with open("whitelist.txt", "r") as file:
    whitelisted_ips = [line.strip() for line in file]

# Read usernames and passwords from a file
with open("user_credentials.txt", "r") as cred_file:
    user_credentials = [line.strip().split(":") for line in cred_file]

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)

class BruteForceLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    ip_address = db.Column(db.String(15), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

with app.app_context():
    db.create_all()
    print("table created successfully")

@limiter.request_filter
def ip_whitelist():
    requesting_ip = get_remote_address()

    # Check if the requested IP is in the whitelist
    return requesting_ip in whitelisted_ips

LOCKOUT_THRESHOLD = 3
LOCKOUT_DURATION = 300  # seconds

IP_BLOCK_THRESHOLD = 5
IP_BLOCK_DURATION = 600  # seconds

def print_detection(message):
    print(f"[Detection] {message}")

@app.route('/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    if request.is_json:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
    else:
        # Handle form data
        username = request.form.get('username')
        password = request.form.get('password')
    
    if not username or not password:
        abort(400, description="Invalid request format. Please provide 'username' and 'password'.")

    # Validate username and password against the list read from the file
    if [username, password] in user_credentials:
        # Redirect to the next page after successful login
        return redirect(url_for('login_successful'))  
    else:
        ip_address = request.remote_addr
        log_failed_attempt(username, ip_address)

        if is_account_locked_out(username):
            return render_template('account_locked.html', message="Account locked out. Try again later.")
        if is_ip_blocked(ip_address):
            return render_template('ip_blocked.html', message="IP blocked. Try again later.")
        if is_anomaly(ip_address):
            return render_template('anomaly_detected.html', message="Anomaly detected. Access denied.")

        return render_template('login_failed.html', message="Login failed")

def log_failed_attempt(username, ip_address):
    log_entry = BruteForceLog(username=username, ip_address=ip_address)
    db.session.add(log_entry)
    db.session.commit()

def is_account_locked_out(username):
    failed_attempts = BruteForceLog.query.filter_by(username=username).order_by(BruteForceLog.timestamp.desc()).limit(LOCKOUT_THRESHOLD).all()

    if len(failed_attempts) >= LOCKOUT_THRESHOLD:
        last_failed_attempt_time = failed_attempts[-1].timestamp
        if (datetime.datetime.utcnow() - last_failed_attempt_time).total_seconds() <= LOCKOUT_DURATION:
            return True

    return False

def is_ip_blocked(ip_address):
    failed_attempts = BruteForceLog.query.filter_by(ip_address=ip_address).order_by(BruteForceLog.timestamp.desc()).limit(IP_BLOCK_THRESHOLD).all()

    if len(failed_attempts) >= IP_BLOCK_THRESHOLD:
        last_failed_attempt_time = failed_attempts[-1].timestamp
        if (datetime.datetime.utcnow() - last_failed_attempt_time).total_seconds() <= IP_BLOCK_DURATION:
            return True

    return False

def is_anomaly(ip_address):
    features = get_failed_attempts_count_features(ip_address)

    if features is not None:
        model = train_anomaly_detection_model()
        features_2d = np.array([[features]]).reshape(1,-1)  # Ensure features is a 2D array
        prediction = model.predict(features_2d)
        if prediction[0] == -1:
            print_detection(f"Anomaly detected for IP: {ip_address}, Failed attempts: {features[0][0]}")  # -1 indicates an anomaly
            return True
    return False

def get_failed_attempts_count_features(ip_address):
    timestamp_limit = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
    failed_attempts = BruteForceLog.query.filter_by(ip_address=ip_address).filter(BruteForceLog.timestamp >= timestamp_limit).count()

    if failed_attempts > 0:
        normalized_failed_attempts = failed_attempts / 5.0  # Adjust normalization factor based on your application
        return [[normalized_failed_attempts]]

    return None

def train_anomaly_detection_model():
    data = [get_failed_attempts_count_features(log.ip_address)[0][0] for log in BruteForceLog.query.all() if log.timestamp >= datetime.datetime.utcnow() - datetime.timedelta(days=7)]
    data = [x for x in data if x is not None]  # Remove None values
    if not data:
        print_detection("No data available for training the model.")
        return None

    # Reshape the data to a 2D array
    data_2d = np.array(data).reshape(-1, 1)

    model = IsolationForest(contamination=0.01)  # Adjust contamination parameter based on your application
    model.fit(data_2d)
    return model

@app.route('/login_successful')
def login_successful():
    return render_template('login_successful.html', message="NO")

@app.route('/')
def index():
    return render_template('login.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    app.run(debug=True)
