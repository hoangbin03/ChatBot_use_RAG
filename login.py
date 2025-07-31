from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from utils import get_db_connection
import hashlib

login_bp = Blueprint('login', __name__)

@login_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_name = request.form['user_name']
        password = request.form['password']
        hashed_password = hashlib.sha256(password.encode()).hexdigest()  

        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT user_id FROM users WHERE user_name=%s AND password=%s", (user_name, hashed_password))
                user = cursor.fetchone()
                print("User từ DB:", user)
                if user:
                    flash("Đăng nhập thành công!", "success")
                    session['user_id'] = user[0] 
                    session['user_name'] = user_name
                    print(user_name)
                    return redirect(url_for('chatbot.home'))  
                else:
                    flash("Sai tên đăng nhập hoặc mật khẩu.", "danger")
        except Exception as e:
            flash(f"Lỗi hệ thống: {e}", "danger")
        finally:
            connection.close()

    return render_template('login.html')

@login_bp.route('/logout')
def logout():
    flash("Bạn đã đăng xuất!", "info")
    return redirect(url_for('login.login'))
