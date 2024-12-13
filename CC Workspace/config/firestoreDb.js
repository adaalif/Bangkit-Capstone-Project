const admin = require('./firebaseAdmin');  // Mengimpor admin yang sudah diinisialisasi
const db = admin.firestore();  // Mendapatkan akses ke Firestore

// Cek apakah Firestore berhasil diakses
db.collection('users')
  .get()
  .then(snapshot => {
    console.log('Firestore connected successfully');
  })
  .catch(err => {
    console.error('Error connecting to Firestore:', err);
  });

module.exports = db; // Mengekspos Firestore db
