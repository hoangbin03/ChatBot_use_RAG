from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, abort
import requests
import os
from datetime import datetime
from functools import wraps
from login import login_bp
from register import register_bp
from Chatbot import chatbot_bp
from embedding import embedding 


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")  

app.register_blueprint(login_bp)
app.register_blueprint(register_bp)
app.register_blueprint(chatbot_bp)
app.register_blueprint(embedding)

@app.route('/')
def index():
    return render_template('login.html')


# Đăng xuất
@app.route('/logout')
def logout():
    session.pop('user_id', None) 
    flash("Đăng xuất thành công!", "success")  
    return redirect(url_for('login.login')) 
 
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
