import pytest
from mainpro import app, db, User, BruteForceLog, train_anomaly_detection_model

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test_brute_force.db'
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client
            db.session.remove()
            db.drop_all()

def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Welcome to the login system!" in response.data

def test_login_successful(client):
    user_credentials = {'username': 'test_user', 'password': 'test_password'}
    with app.app_context():
        db.session.add(User(username=user_credentials['username'], password=user_credentials['password']))
        db.session.commit()

    response = client.post('/login', json=user_credentials)
    assert response.status_code == 200
    assert b"NO" in response.data  # Assuming 'NO' is present in the login_successful.html

def test_login_failed(client):
    user_credentials = {'username': 'test_user', 'password': 'test_password'}
    response = client.post('/login', json=user_credentials)
    assert response.status_code == 200
    assert b"Login failed" in response.data  # Assuming 'Login failed' is present in the login_failed.html

def test_account_locked_out(client):
    user_credentials = {'username': 'locked_user', 'password': 'test_password'}
    with app.app_context():
        for _ in range(4):  # Exceed LOCKOUT_THRESHOLD
            response = client.post('/login', json=user_credentials)
            assert response.status_code == 200

        response = client.post('/login', json=user_credentials)
        assert response.status_code == 403
        assert b"Account locked out. Try again later." in response.data

def test_ip_blocked(client):
    ip_address = '127.0.0.2'
    user_credentials = {'username': 'test_user', 'password': 'test_password'}
    with app.app_context():
        for _ in range(6):  # Exceed IP_BLOCK_THRESHOLD
            response = client.post('/login', json=user_credentials, environ_base={'REMOTE_ADDR': ip_address})
            assert response.status_code == 200

        response = client.post('/login', json=user_credentials, environ_base={'REMOTE_ADDR': ip_address})
        assert response.status_code == 403
        assert b"IP blocked. Try again later." in response.data

def test_anomaly_detected(client):
    ip_address = '127.0.0.3'
    user_credentials = {'username': 'test_user', 'password': 'test_password'}
    with app.app_context():
        # Create an anomaly in the system
        for _ in range(6):
            response = client.post('/login', json=user_credentials, environ_base={'REMOTE_ADDR': ip_address})
            assert response.status_code == 200

        # Train the anomaly detection model with known data
        model = train_anomaly_detection_model()

        # Use an unseen IP address, triggering an anomaly
        response = client.post('/login', json=user_credentials, environ_base={'REMOTE_ADDR': '127.0.0.4'})
        assert response.status_code == 403
        assert b"Anomaly detected. Access denied." in response.data
