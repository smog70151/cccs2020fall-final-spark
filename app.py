from flask import Flask, session, redirect, render_template, request, flash, url_for, json
from flask_socketio import SocketIO, emit, join_room
from flask_login import UserMixin, LoginManager, login_required, current_user, login_user, logout_user
from functools import wraps
from PIL import Image
from datetime import datetime
import requests
import base64
import os
import uuid
import io
from flask import jsonify
from inference import sparkAD

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = 'super secret string'  # Change this!
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.session_protection = "strong"
login_manager.login_view = "login"
login_manager.login_message = "Please LOG IN"
login_manager.login_message_category = "info"

socketio = SocketIO(app)
async_mode = "eventlet"

@app.route('/get_ads', methods=['POST'])
def get_ads():
    print ('Form: ', request.form.to_dict())
    print ('Data: ', request.data)
    print ('Values: ', request.values)
    msgs = request.form.to_dict()
    ad = sparkAD()
    res = ad.preprocess(msgs)
    print (res)
    return jsonify({'code': 0, 'status': 'running', 'data': res['prediction']})

if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.jinja_env.auto_reload = True
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)
