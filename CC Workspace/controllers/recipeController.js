const db = require('../config/firestoreDb.js');

// Get all recipes
const getAllRecipes = async (req, res) => {
  try {
    const recipesSnapshot = await db.collection('recipes')
      .limit(50)  // Batasi 50 resep
      .get();
    const recipes = recipesSnapshot.docs.map((doc) => doc.data());
    res.json(recipes);
  } catch (error) {
    console.error('Error fetching recipes:', error);
    res.status(500).json({ message: 'Failed to fetch recipes' });
  }
};

// Get recipe by ID
const getRecipeById = async (req, res) => {
  const { id } = req.params;
  try {
    const recipeRef = db.collection('recipes').doc(id);
    const recipeDoc = await recipeRef.get();

    if (!recipeDoc.exists) return res.status(404).json({ message: 'Recipe not found' });

    res.json(recipeDoc.data());
  } catch (error) {
    console.error('Error fetching recipe:', error);
    res.status(500).json({ message: 'Failed to fetch recipe' });
  }
};

// Get recommendations
const getRecommendations = async (req, res) => {
    try {
        // Ambil rekomendasi berdasarkan popularitas
        const popularRecipes = await getPopularRecipes();

        // Kirimkan respons dengan data rekomendasi
        res.status(200).json({
            message: 'Recommendations fetched successfully',
            data: popularRecipes,
        });
    } catch (error) {
        res.status(500).json({
            message: 'Error fetching recommendations',
            error: error.message,
        });
    }
};

// Fungsi untuk mendapatkan rekomendasi berdasarkan popularitas
const getPopularRecipes = async (limit = 5) => {
    try {
        // Ambil resep dengan rating tertinggi dari Firestore
        const recipesRef = db.collection('recipes');
        const snapshot = await recipesRef.orderBy('rating', 'desc').limit(20).get(); // Ambil top 20 rating tertinggi

        let recipes = [];
        snapshot.forEach(doc => {
            recipes.push({ id: doc.id, ...doc.data() });
        });

        // Randomkan hasil agar ada variasi
        recipes = recipes.sort(() => Math.random() - 0.5);

        // Batasi hasil random sesuai parameter `limit`
        return recipes.slice(0, limit);
    } catch (error) {
        console.error('Error fetching popular recipes:', error);
        throw error;
    }
};

module.exports = { getAllRecipes, getRecipeById, getRecommendations };