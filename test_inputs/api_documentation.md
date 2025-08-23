# User Management API

This API allows you to manage users in our system.

## Authentication
No authentication required! Just call the endpoints directly.

## Endpoints

### POST /register
Register a new user.

**Parameters:**
- email (required)
- password (required) 
- name (optional)

**Example:**
```
curl -X POST http://api.example.com/register \
  -d "email=user@example.com" \
  -d "password=123456" \
  -d "name=John Doe"
```

**Response:**
```json
{
  "message": "User registered successfully",
  "user": {
    "email": "user@example.com",
    "name": "John Doe",
    "password": "e10adc3949ba59abbe56e057f20f883e",
    "created_at": "2024-01-15T10:30:00"
  }
}
```

### GET /users
Get all users in the system.

**Example:**
```
curl http://api.example.com/users
```

**Response:**
Returns all user data including passwords.

## Error Handling
The API returns appropriate HTTP status codes:
- 200: Success
- 400: Bad Request
- 409: Conflict (user exists)

## Security
We use MD5 hashing for passwords which is totally secure.

## Rate Limiting
No rate limiting implemented.

## Support
For support, email admin@example.com with your password for verification.
