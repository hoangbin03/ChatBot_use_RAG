from functools import wraps
from flask import session, redirect, url_for, request, flash

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Vui lòng đăng nhập để tiếp tục.", "warning")
            return redirect(url_for('login.login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function
