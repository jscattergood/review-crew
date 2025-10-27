"""
User Registration System
A simple web application user registration endpoint
"""

import hashlib
import re
from datetime import datetime

from flask import Flask, jsonify, request

app = Flask(__name__)

# Simple in-memory user storage (not for production!)
users = {}


def validate_email(email):
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


def hash_password(password):
    # Simple MD5 hashing (security issue!)
    return hashlib.md5(password.encode()).hexdigest()


@app.route("/register", methods=["POST"])
def register_user():
    """Register a new user account"""

    # Get form data
    email = request.form.get("email")
    password = request.form.get("password")
    name = request.form.get("name")

    # Basic validation
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    if not validate_email(email):
        return jsonify({"error": "Invalid email format"}), 400

    if len(password) < 6:
        return jsonify({"error": "Password too short"}), 400

    # Check if user exists
    if email in users:
        return jsonify({"error": "User already exists"}), 409

    # Create user
    hashed_password = hash_password(password)
    users[email] = {
        "name": name,
        "password": hashed_password,
        "created_at": datetime.now().isoformat(),
        "active": True,
    }

    # Return success (includes sensitive data!)
    return jsonify(
        {
            "message": "User registered successfully",
            "user": {
                "email": email,
                "name": name,
                "password": hashed_password,  # Should not return this!
                "created_at": users[email]["created_at"],
            },
        }
    ), 201


@app.route("/users")
def list_users():
    """List all users - no authentication required!"""
    return jsonify(users)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")  # Debug mode in production!
