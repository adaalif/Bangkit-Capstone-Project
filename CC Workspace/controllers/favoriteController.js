const db = require('../config/firestoreDb.js');

// Mendapatkan daftar resep favorit user
exports.getFavorites = async (req, res) => {
  try {
    const { userId } = req.body; // Mengambil userId hanya dari body
    console.log('Getting favorites for userId:', userId);

    if (!userId) {
      return res.status(400).json({ 
        success: false, 
        message: 'User ID is required in request body' 
      });
    }

    const snapshot = await db.collection('favorites')
      .where('userId', '==', userId)
      .get();

    console.log('Found favorites:', snapshot.size);

    // Jika tidak ada favorit, kembalikan array kosong
    if (snapshot.empty) {
      return res.status(200).json({ 
        success: true, 
        data: [],
        message: 'No favorites found for this user'
      });
    }

    // Ambil detail resep untuk setiap favorit
    const favoritesPromises = snapshot.docs.map(async (doc) => {
      const favoriteData = doc.data();
      const recipeDoc = await db.collection('recipes').doc(favoriteData.recipeId).get();
      
      return {
        id: doc.id,
        userId: favoriteData.userId,
        recipeId: favoriteData.recipeId,
        createdAt: favoriteData.createdAt,
        recipe: recipeDoc.exists ? recipeDoc.data() : null
      };
    });

    const favorites = await Promise.all(favoritesPromises);
    const validFavorites = favorites.filter(fav => fav.recipe !== null);

    res.status(200).json({
      success: true,
      data: validFavorites,
      message: validFavorites.length > 0 ? 'Favorites retrieved successfully' : 'No valid favorites found'
    });

  } catch (error) {
    console.error('Error getting favorites:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Failed to get favorites',
      error: error.message
    });
  }
};

// Menambahkan resep ke daftar favorit
exports.addFavorite = async (req, res) => {
  try {
    const { userId, recipeId } = req.body;

    console.log('Debug values:', {
      userId,
      recipeId,
      body: req.body
    });

    if (!userId || !recipeId) {
      return res.status(400).json({ 
        success: false, 
        message: 'User ID and Recipe ID are required in request body.',
        debug: {
          providedUserId: userId,
          providedRecipeId: recipeId
        }
      });
    }

    // Coba berbagai format ID resep
    let recipeDoc;
    let finalRecipeId = recipeId;
    const possibleIds = [
      recipeId,
      recipeId.replace('recipe', ''),
      `recipe_${recipeId.replace('recipe', '')}`,
      recipeId.replace('_', '')
    ];

    console.log('Trying possible recipe IDs:', possibleIds);

    for (const id of possibleIds) {
      recipeDoc = await db.collection('recipes').doc(id).get();
      if (recipeDoc.exists) {
        finalRecipeId = id;
        break;
      }
    }

    if (!recipeDoc.exists) {
      // List semua dokumen di collection recipes untuk debugging
      const allRecipes = await db.collection('recipes').limit(5).get();
      const sampleRecipes = [];
      allRecipes.forEach(doc => {
        sampleRecipes.push(doc.id);
      });

      return res.status(404).json({ 
        success: false, 
        message: 'Recipe not found.',
        debug: {
          triedIds: possibleIds,
          sampleExistingIds: sampleRecipes // Menampilkan beberapa ID resep yang ada di database
        }
      });
    }

    // Cek apakah sudah ada di favorit
    const snapshot = await db.collection('favorites')
      .where('userId', '==', userId)
      .where('recipeId', '==', finalRecipeId)
      .get();

    if (!snapshot.empty) {
      return res.status(400).json({ 
        success: false, 
        message: 'Recipe already exists in favorites.' 
      });
    }

    // Tambahkan ke favorit
    const favoriteData = {
      userId,
      recipeId: finalRecipeId,
      createdAt: new Date(),
      recipeName: recipeDoc.data().title || recipeDoc.data().name || ''
    };

    const newFavoriteRef = await db.collection('favorites').add(favoriteData);

    res.status(201).json({
      success: true,
      data: { 
        id: newFavoriteRef.id,
        ...favoriteData
      },
      message: 'Recipe successfully added to favorites.'
    });
  } catch (error) {
    console.error('Detailed error in addFavorite:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Failed to add recipe to favorites.',
      error: error.message,
      debug: {
        errorDetails: error.toString()
      }
    });
  }
};

// Menghapus resep dari daftar favorit
exports.removeFavorite = async (req, res) => {
  try {
    const { userId, recipeId } = req.body;

    console.log('Debug delete:', {
      recipeId,
      userId
    });

    if (!userId || !recipeId) {
      return res.status(400).json({ 
        success: false, 
        message: 'User ID and Recipe ID are required in request body.' 
      });
    }

    // Coba berbagai format ID resep
    let finalRecipeId = recipeId;
    const possibleIds = [
      recipeId,
      recipeId.replace('recipe', ''),
      `recipe_${recipeId.replace('recipe', '')}`,
      recipeId.replace('_', '')
    ];

    console.log('Trying possible recipe IDs for deletion:', possibleIds);

    let foundFavorite = false;
    let snapshotToDelete;

    // Coba setiap kemungkinan format ID
    for (const id of possibleIds) {
      const snapshot = await db.collection('favorites')
        .where('userId', '==', userId)
        .where('recipeId', '==', id)
        .get();

      if (!snapshot.empty) {
        foundFavorite = true;
        snapshotToDelete = snapshot;
        finalRecipeId = id;
        break;
      }
    }

    if (!foundFavorite) {
      // Ambil beberapa favorit yang ada untuk debugging
      const allFavorites = await db.collection('favorites')
        .where('userId', '==', userId)
        .limit(5)
        .get();
      
      const existingFavorites = [];
      allFavorites.forEach(doc => {
        existingFavorites.push(doc.data().recipeId);
      });

      return res.status(404).json({ 
        success: false, 
        message: 'Recipe not found in favorites.',
        debug: {
          triedIds: possibleIds,
          userFavorites: existingFavorites // Menampilkan beberapa favorit yang dimiliki user
        }
      });
    }

    // Hapus dokumen favorit
    const docId = snapshotToDelete.docs[0].id;
    await db.collection('favorites').doc(docId).delete();

    res.status(200).json({ 
      success: true, 
      message: 'Recipe successfully removed from favorites.',
      data: {
        userId,
        recipeId: finalRecipeId
      }
    });
  } catch (error) {
    console.error('Error removing favorite:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Failed to remove recipe from favorites.',
      error: error.message 
    });
  }
};

// Mendapatkan kategori dari resep favorit
exports.getFavoriteCategories = async (req, res) => {
  try {
    const { userId } = req.body; // Mengambil userId dari body request

    if (!userId) {
      return res.status(400).json({ 
        success: false, 
        message: 'User ID is required in request body' 
      });
    }

    const snapshot = await db.collection('favorites')
      .where('userId', '==', userId)
      .get();

    if (snapshot.empty) {
      return res.status(200).json({ 
        success: true, 
        data: [],
        message: 'No favorites found for this user'
      });
    }

    // Ambil detail resep untuk mendapatkan kategorinya
    const categoriesPromises = snapshot.docs.map(async (doc) => {
      const favoriteData = doc.data();
      const recipeDoc = await db.collection('recipes').doc(favoriteData.recipeId).get();
      
      if (!recipeDoc.exists) return null;
      
      const recipeData = recipeDoc.data();
      return recipeData.category || recipeData.categories || [];
    });

    const categories = await Promise.all(categoriesPromises);
    
    // Flatten array dan hilangkan null/undefined
    const flatCategories = categories
      .filter(cat => cat !== null)
      .flat()
      .filter(cat => cat); // Hilangkan nilai kosong

    // Hilangkan duplikat
    const uniqueCategories = [...new Set(flatCategories)];

    res.status(200).json({ 
      success: true, 
      data: uniqueCategories,
      message: uniqueCategories.length > 0 ? 'Categories retrieved successfully' : 'No categories found'
    });
  } catch (error) {
    console.error('Error getting favorite categories:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Failed to get favorite categories',
      error: error.message
    });
  }
};
