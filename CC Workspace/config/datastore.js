const admin = require('firebase-admin');
const serviceAccount = require('./service.json');

// Inisialisasi Firebase Admin SDK
if (admin.apps.length === 0) {
  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
  });
}

const db = admin.firestore();

// Fungsi untuk memasukkan data ke Firestore
const insertDataToFirestore = async (collectionName, dataset) => {
  try {
    if (Array.isArray(dataset)) {
      const batch = db.batch();
      dataset.forEach((item) => {
        const docRef = db.collection(collectionName).doc();
        batch.set(docRef, item);
      });
      await batch.commit();
      console.log(`Dataset berhasil ditambahkan ke koleksi "${collectionName}"`);
    } else {
      const docRef = db.collection(collectionName).doc();
      await docRef.set(dataset);
      console.log(`Data berhasil ditambahkan ke koleksi "${collectionName}"`);
    }
  } catch (error) {
    console.error('Error saat memasukkan data ke Firestore:', error);
  }
};

module.exports = { db, insertDataToFirestore }; // Mengekspos db dan fungsi insertDataToFirestore
