const admin = require('firebase-admin');
const serviceAccount = require('./service.json');
// Inisialisasi Firebase Admin SDK hanya jika belum ada aplikasi yang diinisialisasi
if (admin.apps.length === 0) {
  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    databaseURL: "https://ingrevia-default-rtdb.asia-southeast1.firebasedatabase.app"  // Pastikan URL sesuai dengan project Firebase Anda
  });
}

module.exports = admin; // Mengekspos admin untuk digunakan di file lain
