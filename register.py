from flask import Blueprint, render_template, request, redirect, url_for, flash
from utils import get_db_connection
import hashlib
import logging
import re

register_bp = Blueprint('register', __name__)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)

@register_bp.route('/register', methods=['GET', 'POST'])
def register(): 
    if request.method == 'POST':
        user_name = request.form['user_name']
        ngay_sinh = request.form['ngay_sinh']
        email = request.form['email']
        sodienthoai = request.form['sodienthoai']
        password = request.form['password']
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE user_name = %s OR email = %s", (user_name, email))
                if cursor.fetchone():
                    flash("Tài khoản hoặc email đã tồn tại, vui lòng thử lại.", "danger")
                    return redirect(url_for('register.register'))

                cursor.execute(
                    "INSERT INTO users (user_name, ngay_sinh, email, sodienthoai, password) VALUES (%s, %s, %s, %s, %s)",
                    (user_name, ngay_sinh, email, sodienthoai, hashed_password)
                )
                connection.commit()
                flash("Tạo tài khoản thành công!", "success")
                return redirect(url_for('login.login'))
        except Exception as e:
            print(f"Lỗi: {e}")
            connection.rollback()
            flash("Đăng ký không thành công, vui lòng thử lại.", "danger")
        finally:
            connection.close()

    return render_template('register.html')
