const db = require('../config/firestoreDb.js');
const axios = require('axios');

// Fungsi untuk mendapatkan rekomendasi berdasarkan popularitas
const getPopularRecipes = async (limit = 5) => {
    try {
        // Ambil resep dengan rating tertinggi dari Firestore
        const recipesRef = db.collection('recipes');
        const snapshot = await recipesRef
            .orderBy('rating', 'desc') // Mengurutkan berdasarkan rating
            .limit(20) // Ambil 20 resep dengan rating tertinggi
            .get();

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

// Fungsi untuk mendapatkan rekomendasi dari ML model
exports.getRecommendations = async (req, res) => {
    try {
        const { weight, height, age, gender, activity_level, liked_recipe_indices } = req.body;

        // Validasi input
        if (!weight || !height || !age || gender === undefined || !activity_level) {
            return res.status(400).json({
                success: false,
                message: 'Semua field (weight, height, age, gender, activity_level) harus diisi'
            });
        }

        // Ambil 50 resep dari Firestore terlebih dahulu
        const recipesRef = db.collection('recipes');
        const snapshot = await recipesRef.limit(50).get();
        
        const recipes = [];
        snapshot.forEach(doc => {
            recipes.push({
                id: doc.id,
                ...doc.data()
            });
        });

        // Setelah mendapatkan resep, kirim ke ML API untuk rekomendasi
        const mlRequestData = {
            weight: Number(weight),
            height: Number(height),
            age: Number(age),
            gender: Number(gender),
            activity_level: Number(activity_level),
            liked_recipe_indices: liked_recipe_indices || [],
            recipes: recipes // Kirim resep yang sudah diambil
        };

        // Hit ML API
        const mlResponse = await axios.post(
            'https://machine-learning-1042086567112.asia-southeast1.run.app/api/recommend',
            mlRequestData
        );

        return res.status(200).json({
            success: true,
            message: 'Rekomendasi resep berhasil didapatkan',
            data: {
                user_metrics: {
                    weight,
                    height,
                    age,
                    gender,
                    activity_level
                },
                recipes: recipes, // 50 resep yang diambil
                recommendations: mlResponse.data // Hasil rekomendasi dari ML
            }
        });

    } catch (error) {
        console.error('Error getting recommendations:', error);
        return res.status(500).json({
            success: false,
            message: 'Gagal mendapatkan rekomendasi',
            error: error.message
        });
    }
};

// Search recipes by ingredients
exports.searchByIngredients = async (req, res) => {
    const { query } = req.body;

    if (!query) {
        return res.status(400).json({
            success: false,
            message: 'Query parameter is required'
        });
    }

    try {
        const searchIngredients = query.toLowerCase().split(',').map(ing => ing.trim());
        const recipesRef = db.collection('recipes');
        const snapshot = await recipesRef.limit(50).get();  // Batasi 50 resep

        let recipes = [];
        snapshot.forEach(doc => {
            const data = doc.data();
            if (data.ingredients && Array.isArray(data.ingredients)) {
                const recipeIngredients = data.ingredients.map(ing => ing.toLowerCase());
                const hasAllIngredients = searchIngredients.every(searchIng => 
                    recipeIngredients.some(recipeIng => recipeIng.includes(searchIng))
                );

                if (hasAllIngredients) {
                    recipes.push({ id: doc.id, ...data });
                }
            }
        });

        // Batasi hasil pencarian jika masih melebihi 50
        if (recipes.length > 50) {
            recipes = recipes.slice(0, 50);
        }

        if (recipes.length === 0) {
            return res.status(404).json({
                success: false,
                message: 'No recipes found containing all the specified ingredients'
            });
        }

        res.status(200).json({
            success: true,
            data: recipes
        });
    } catch (error) {
        console.error('Search error:', error);
        res.status(500).json({
            success: false,
            message: 'Internal server error'
        });
    }
};

// Search recipes by title
exports.searchByTitle = async (req, res) => {
    const { query } = req.body;

    if (!query) {
        return res.status(400).json({
            success: false,
            message: 'Query parameter is required'
        });
    }

    try {
        const recipesRef = db.collection('recipes');
        const snapshot = await recipesRef.limit(50).get();  // Batasi 50 resep

        let recipes = [];
        snapshot.forEach(doc => {
            const recipeData = doc.data();
            if (recipeData.title && 
                recipeData.title.toLowerCase().includes(query.toLowerCase())) {
                recipes.push({ id: doc.id, ...recipeData });
            }
        });

        // Batasi hasil pencarian jika masih melebihi 50
        if (recipes.length > 50) {
            recipes = recipes.slice(0, 50);
        }

        if (recipes.length === 0) {
            return res.status(404).json({
                success: false,
                message: 'No recipes found with the given title'
            });
        }

        res.status(200).json({
            success: true,
            data: recipes
        });
    } catch (error) {
        console.error('Search error:', error);
        res.status(500).json({
            success: false,
            message: 'Internal server error'
        });
    }
};

// Get categories
exports.getCategories = async (req, res) => {
    try {
        const { query } = req.body;
        
        if (!query) {
            // Jika tidak ada query, kembalikan semua kategori yang terorganisir
            const categories = {
                protein: [
                    'Chicken', 'Beef', 'Lamb', 'Shrimp', 'Squid', 'Crab', 
                    'Tofu', 'Tempe', 'Egg', 'Fish'
                ],
                carbs: [
                    'Oat', 'Rice', 'Corn', 'Potato', 'Cassava', 
                    'Bread', 'Flour', 'Sago'
                ],
                vegetables: [
                    'Spinach', 'Broccoli', 'Carrot', 'Bean', 'Tomato',
                    'Cucumber', 'Cabbage', 'Eggplant'
                ],
                dairy_and_condiments: [
                    'Milk', 'Cheese', 'Butter', 'Margarine', 'Yogurt',
                    'Mayonnaise', 'Soy sauce', 'Oyster sauce', 'Chili',
                    'Coconut milk', 'Vinegar', 'Shrimp paste'
                ],
                spices_and_herbs: [
                    'Garlic', 'Onion', 'Chili', 'Curry', 'Coriander',
                    'Lemongrass', 'Ginger'
                ]
            };

            return res.status(200).json({ 
                success: true, 
                data: categories 
            });
        }

        // Split query menjadi array dan bersihkan dari spasi
        const searchTerms = query.split(',').map(term => term.trim());
        
        const recipesRef = db.collection('recipes');
        const snapshot = await recipesRef.limit(50).get();
  
        if (snapshot.empty) {
            return res.status(404).json({ 
                success: false, 
                message: 'No recipes found' 
            });
        }
  
        const matchingRecipes = [];
  
        snapshot.forEach((doc) => {
            const data = doc.data();
            
            // Periksa field 'categories' bukan 'category'
            const hasMatchingCategory = Array.isArray(data.categories) && 
                searchTerms.some(term => 
                    data.categories.some(cat => 
                        cat.toLowerCase().includes(term.toLowerCase())
                    )
                );

            if (hasMatchingCategory) {
                matchingRecipes.push({ id: doc.id, ...data });
            }
        });
  
        if (matchingRecipes.length === 0) {
            return res.status(404).json({ 
                success: false, 
                message: `No recipes found for categories: ${searchTerms.join(', ')}` 
            });
        }
  
        res.status(200).json({ success: true, data: matchingRecipes });
    } catch (error) {
        console.error('Error getting categories:', error);
        res.status(500).json({ 
            success: false, 
            message: 'Internal server error' 
        });
    }
};
  
  
// Discover new recipes
exports.discover = async (req, res) => {
    try {
        // Ambil resep terbaru berdasarkan rating tertinggi
        const recipesRef = db.collection('recipes');
        const snapshot = await recipesRef
            .orderBy('rating', 'desc') // Mengurutkan berdasarkan rating tertinggi
            .limit(50) // Ambil 5 resep dengan rating tertinggi
            .get();

        let recipes = [];
        snapshot.forEach(doc => {
            recipes.push({ id: doc.id, ...doc.data() });
        });

        if (recipes.length === 0) {
            return res.status(404).send('No new recipes discovered');
        }

        // Kirimkan hasil penemuan resep
        res.status(200).json({
            message: 'Recipes Today discovered successfully',
            data: recipes,
        });
    } catch (error) {
        console.error('Error discovering new recipes:', error);
        res.status(500).json({
            message: 'Error discovering new recipes',
            error: error.message,
        });
    }
};
