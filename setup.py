from datetime import datetime
from app.main import socketio, app

if __name__ == "__main__":
    socketio.run(app, debug=True)
