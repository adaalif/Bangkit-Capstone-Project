const admin = require('firebase-admin'); // Mengimpor Firebase Admin SDK
const db = require('../config/firestoreDb.js');

// Middleware untuk autentikasi pengguna
const authenticate = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ success: false, message: 'Authorization header is missing or invalid.' });
    }

    const token = authHeader.split(' ')[1];

    if (!token) {
      return res.status(401).json({ success: false, message: 'Token is missing.' });
    }

    const decodedToken = await admin.auth().verifyIdToken(token);

    // Tambahkan informasi user ke dalam request
    req.user = {
      id: decodedToken.uid, // Pastikan `uid` tersedia di token Firebase
      email: decodedToken.email || null,
      name: decodedToken.name || null,
    };

    next(); // Lanjutkan ke handler berikutnya
  } catch (error) {
    console.error('Authentication error:', error);
    return res.status(401).json({ success: false, message: 'Failed to authenticate token.' });
  }
};
module.exports = authenticate;
