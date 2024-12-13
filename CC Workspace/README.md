# Ingrevia Cloud Computing Workspace!
Backend API untuk aplikasi Ingrevia, We will available soon!

## Cloud Architecture
![Ingrevia Cloud Architecture Fixed.jpg](https://github.com/fahrezi93/ingrevia-project/blob/db41ff0637a2dc0a5dfa0565bd9388723ffd50b2/Ingrevia%20Cloud%20Architecture%20Fixed%20(2).jpg)

## URL API Machine Learning Model
```bash
https://machine-learning-1042086567112.asia-southeast1.run.app
```

### 1. **Register User**

**URL:** `POST /auth/register`  
**Endpoint:** [http://localhost:5000/auth/login/register]
**Request:**

```json
{
  "email": "user@gmail.com",
  "name" : "newuser",
  "password": "321"
}
```

**Response Berhasil:**

```json
{
    "success": true,
    "message": "User berhasil didaftarkan.",
    "data": {
        "userId": "9MzkwGhXhTcnQo7dyfO82UAGPNA3",
        "email": "contoh3@gmail.com",
        "name": "contohi"
    }
}

**Response Gagal:**
{
    "success": false,
    "message": "Email sudah terdaftar."
}

```

### 2. **Login User Email**

**URL:** `POST /auth/login-user`  
**Endpoint:** [http://localhost:5000/auth/login/email]

**Request:**

```json
{
  "email": "user@example.com",
  "password": "userpassword"
}
```

**Response Berhasil:**

```json
{
    "success": true,
    "message": "Login berhasil",
    "data": {
        "token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJodHRwczovL2lkZW50aXR5dG9vbGtpdC5nb29nbGVhcGlzLmNvbS9nb29nbGUuaWRlbnRpdHkuaWRlbnRpdHl0b29sa2l0LnYxLklkZW50aXR5VG9vbGtpdCIsImlhdCI6MTczNDAxMzM2MSwiZXhwIjoxNzM0MDE2OTYxLCJpc3MiOiJmaXJlYmFzZS1hZG1pbnNkay1uNGFvbkBpbmdyZXZpYS5pYW0uZ3NlcnZpY2VhY2NvdW50LmNvbSIsInN1YiI6ImZpcmViYXNlLWFkbWluc2RrLW40YW9uQGluZ3JldmlhLmlhbS5nc2VydmljZWFjY291bnQuY29tIiwidWlkIjoidm8zb0ZiYWZhQ2cydnlPT1RBNHJTSGExdUhrMSJ9.J-CWFRgeWQVKWn3hEc2wPr4eOqhdbu2xwMBxjqgT7nzJwe0aHdXpa5GkamdcFiVrBUjRd_IHCfX33sxMEvSA3pcflpDQf7LLO6hXlalIJaMNy-fZaNBJB71AM2x2IwSIGuwegG1y2C_eEoWo8DFGQWF7zKMUZCd3_hVfbCF6_H7OwN3r7ZkEwBuQEWO_wmWtvicDCYAebKldLBfQ_uWvHm2EcqvqKO1Zz_K0COev4VzS4XEn2FGYknFoR5cwJyyZzieiMMHCWkBgM17NC9LDTpMSD_yH-oitJy57wWl3H2iuwxQjPattkSu-DjjnPIebq9qla0apoa7r9h1_7f5JcA",
        "user": {
            "email": "mohfahrezi93@gmail.com",
            "displayName": "Fahrezi"
        }
    }
}
```

**Response Gagal:**

```json
{
    "success": false,
    "message": "Email atau password salah"
}
```

**Response Gagal 3 kali percobaan login dalam 5 menit:**
```json
{
    "success": false,
    "message": "Terlalu banyak percobaan login gagal. Link reset password telah dikirim ke email Anda.",
    "resetLink": "https://ingrevia.firebaseapp.com/__/auth/action?mode=resetPassword&oobCode=DQeXvpFY26NnN5MmWVCKI0emB5ZskdnXNVSX-aCXsjIAAAGTu0aq6w&apiKey=AIzaSyAhf6r4vbQ5FVFX_-jQiu3lasHVdJK-1GQ&lang=en"
}
  
```
## 3. **Forget Password**

**URL:** `POST /auth/forget-password`  
**Endpoint:** [http://localhost:5000/auth/forget-password]

**Request:**

```json
{
  "email": "user@example.com"
}
```

**Response:**

```json
{
    "success": true,
    "message": "Link reset password telah dikirim ke email Anda",
    "data": {
        "email": "user@example.com",
        "resetLink": "https://ingrevia.firebaseapp.com/__/auth/action?mode=resetPassword&oobCode=8ChfSoPpUYdrbsWXRfL7LFFc5CUfTVxUOID9KhNouikAAAGTutXstA&apiKey=AIzaSyAhf6r4vbQ5FVFX_-jQiu3lasHVdJK-1GQ&lang=en"
    }
}
```

**Response Gagal:**

```json
{
    "success": false,
    "message": "Email tidak terdaftar"
}
```
## 4. **Login User Google**

**URL:** `POST /auth/login-google`  
**Endpoint:** [http://localhost:5000/auth/login/google] 

**Request:**

```json
{
  "idToken": "idToken"
}
``` 

**Response:**

```json
{
    "success": true,
    "message": "Login berhasil",
    "data": {
        "token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJHZWV4cFVTQWlkaEtqRXVnenhTZSIsImVtYWlsIjoidXNlckBleGFtcGxlLmNvbSIsImlhdCI6MTczMzg5MjI2OCwiZXhwIjoxNzMzODk1ODY4fQ.CveHfCMNwpKGhDmtCrX2PlUbkjk6WZyBph5u_GT5M7w"
    }
}
```

**Response Gagal:**

```json
{
    "success": false,
    "message": "Login gagal"
}
```

## 5. **Logout User**

**URL:** `POST /auth/logout`  
**Endpoint:** [http://localhost:5000/auth/logout]

**Request:**

```json
{
    "email" : "mohfahrezi93@gmail.com"
}
```
**Response Berhasil:**
```json
{
    "message": "Logout berhasil"
}
```
**Response Gagal:**
```json
{
    "success": false,
    "message": "Logout gagal"
}
```
## 6. **Get User**

**URL:** `GET /home/user`  
**Endpoint:** [http://localhost:5000/home/user]

**Request:**

```json
{
  "userId": "6d4cfb4d-f362-4dc7-941d-05804626d299"
}
```

**Response Berhasil:**

```json
{
    "success": true,
    "data": {
        "userId": "6d4cfb4d-f362-4dc7-941d-05804626d299",
        "createdAt": {
            "_seconds": 1733982500,
            "_nanoseconds": 125000000
        },
        "name": "Fahrezi Lengkap",
        "profilePicUrl": "https://example.com/images/profile.jpg",
        "updatedAt": "2024-12-12T09:17:36.062Z",
        "email": "mohfahrezi93@gmail.com"
    }
}
```

**Response Gagal:**

```json
{
    "message": "User not found"
}
```

# 7. **Update User**

**URL:** `PUT /home/user`  
**Endpoint:** [http://localhost:5000/home/user]

**Request:**

```json
{
    "userId": "6d4cfb4d-f362-4dc7-941d-05804626d299",
    "name": "Fahrezi Lengkap",
    "profilePicUrl": "https://example.com/images/profile.jpg"
}
```

**Response Berhasil:**

```json
{
    "success": true,
    "message": "Profile updated successfully",
    "data": {
        "name": "Fahrezi Lengkap",
        "email": "mohfahrezi93@gmail.com"
    }
}
```

**Response Gagal:**

```json
{
    "success": false,
    "message": "User not found"
}
```
# 8. **Update Profile Picture**

**URL:** `PUT /home/user/profile-pic`  
**Endpoint:** [http://localhost:5000/home/user/profile-pic]

**Request:**

```json
{
    "userId": "6d4cfb4d-f362-4dc7-941d-05804626d299",
    "profilePicUrl": "https://example.com/images/profile.jpg"
}
```

**Response Berhasil:**

```json
{
    "success": true,
    "message": "Profile picture updated successfully",
    "data": {
        "profilePicUrl": "https://example.com/images/profile.jpg",
        "updatedAt": "2024-12-12T09:17:36.062Z"
    }
}
```

**Response Gagal:**

```json
{
    "success": false,
    "message": "User not found"
}
```
# 9. **Get Recommendations**

**URL:** `GET /home/recommendations`  
**Endpoint:** [http://localhost:5000/home/recommendations]

**Request:**

```json
{}
```
# 10. **Search**

**URL:** `GET /home/search`  
**Endpoint:** [http://localhost:5000/home/search?query=Sugar]

**Request:**

```json
{
    "query": "Sugar"
}
```

**Response Berhasil:**
```json
{
    "success": true,
    "message": "Search berhasil",
    "data": []
}
```
**Response Gagal:**

```json
{
    "success": false,
    "message": "Search gagal"
}
```

# 11. **Get Categories**

**URL:** `GET /home/categories`  
**Endpoint:** [http://localhost:5000/home/categories]

**Request:**

```json
{
  "query": "Chicken"
}
```

**Response Berhasil:**

```json
{
    "success": true,
    "message": "Categories berhasil diambil",
    "data": []
}
```

**Response Gagal:**

```json
{
    "success": false,
    "message": "Categories gagal diambil"
}
```

# 12. **Get Discover**

**URL:** `GET /home/discover`  
**Endpoint:** [http://localhost:5000/home/discover]

**Request:**

```json
{}
```

**Response Berhasil:**

```json
{
    "success": true,
    "message": "Discover berhasil diambil",
    "data": []
}
```

**Response Gagal:**

```json
{
    "success": false,
    "message": "Discover gagal diambil"
}
```
  # 13. **Get Recipes**

**URL:** `GET /recipes`  
**Endpoint:** [http://localhost:5000/recipes]

**Request:**

```json
{}
``` 

**Response Berhasil:**  

```json
{
    "success": true,
    "message": "Recipes berhasil diambil",
    "data": []
}
``` 

**Response Gagal:**

```json
{
    "success": false,
    "message": "Recipes gagal diambil"
}
``` 

# 14. **Get Recipe By Id**

**URL:** `GET /recipes/:id`  
**Endpoint:** [http://localhost:5000/recipes/recipe_568]

**Request:**

```json
{}
```

**Response Berhasil:**

```json
{
    "success": true,
    "message": "Recipe berhasil diambil",
    "data": []
}
```

**Response Gagal:**

```json
{
    "success": false,
    "message": "Recipe gagal diambil"
}
```

# 15. **Get Favorite**

**URL:** `GET /favorite`  
**Endpoint:** [http://localhost:5000/favorite]

**Request:**

```json
{}
```
**Response Berhasil:**

```json
{
    "success": true,
    "message": "Recipe berhasil diambil",
    "data": []
}
```
**Response Gagal:**

```json
{
    "success": false,
    "message": "Recipe gagal diambil"
}
```
# 16. **Add Favorite**

**URL:** `POST /favorite/addfavorite`  
**Endpoint:** [http://localhost:5000/favorite/addfavorite]

**Request:**

```json
{
  "userId": "6d4cfb4d-f362-4dc7-941d-05804626d299",
  "recipeId": "recipe554"
}
```

**Response Berhasil:**

```json
{
    "success": true,
    "data": {
        "id": "flVSylfSddmEynb2Xfqk",
        "userId": "6d4cfb4d-f362-4dc7-941d-05804626d299",
        "recipeId": "recipe_554",
        "createdAt": "2024-12-12T10:38:12.410Z",
        "recipeName": "Penne with Wilted Arugula, Radicchio, and Smoked Mozzarella"
    },
    "message": "Recipe successfully added to favorites."
}
```
**Response Gagal:**

```json
{
    "success": false,
    "message": "Recipe gagal ditambahkan ke favorit"
}
```

# 17. **Delete Favorite**

**URL:** `DELETE /favorite/deletefavorite`  
**Endpoint:** [http://localhost:5000/favorite/deletefavorite]

**Request:**

```json
{
  "userId": "6d4cfb4d-f362-4dc7-941d-05804626d299",
  "recipeId": "recipe554"
}
```

**Response Berhasil:**

```json
{
    "success": true,
    "message": "Recipe successfully removed from favorites.",
    "data": {
        "userId": "6d4cfb4d-f362-4dc7-941d-05804626d299",
        "recipeId": "recipe_554"
    }
}
```
**Response Gagal:**

```json
{
    "success": false,
    "message": "Recipe gagal dihapus dari favorit"
}
```
# 18. **Get Favorite Categories**

**URL:** `GET /favorite/categories`  
**Endpoint:** [http://localhost:5000/favorite/categories]

**Request:**

```json
{
  "userId": "6d4cfb4d-f362-4dc7-941d-05804626d299",
  "recipeId": "recipe86"
}
``` 

**Response Berhasil:**

```json
{
    "success": true,
    "message": "Categories berhasil diambil",
    "data": []
}
```
  
**Response Gagal:**

```json
{
    "success": false,
    "message": "Categories gagal diambil"
}
```

