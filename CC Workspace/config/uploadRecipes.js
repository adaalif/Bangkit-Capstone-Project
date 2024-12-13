const Firestore = require('@google-cloud/firestore');
const fs = require('fs');
const path = require('path');

const db = new Firestore({
  projectId: 'ingrevia',
  keyFilename: path.resolve('../firebaseService.json'),
});

const filePath = path.resolve('./resep.json');

fs.readFile(filePath, 'utf8', async (err, data) => {
  if (err) {
    console.error('Error reading file:', err);
    return;
  }

  try {
    const recipes = JSON.parse(data);

    if (!Array.isArray(recipes)) {
      throw new Error('File JSON tidak valid, data harus berupa array.');
    }

    const batchSize = 100; // Maksimum dokumen per batch
    for (let i = 0; i < recipes.length; i += batchSize) {
      const batch = db.batch();
      const chunk = recipes.slice(i, i + batchSize);

      chunk.forEach((recipe, idx) => {
        const docRef = db.collection('recipes').doc(`recipe_${i + idx + 1}`);
        batch.set(docRef, recipe);
      });

      await batch.commit();
      console.log(`Batch ${Math.floor(i / batchSize) + 1} berhasil disimpan.`);
    }
    console.log('Semua data berhasil disimpan ke Firestore.');
  } catch (error) {
    console.error('Error:', error);
  }
});
