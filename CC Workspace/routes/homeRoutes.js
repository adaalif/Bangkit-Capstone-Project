const express = require('express');
const router = express.Router();
const homeController = require('../controllers/homeController');

// Home endpoints
router.post('/recommendations', homeController.getRecommendations); // Rekomendasi resep berdasarkan popularitas
router.post('/search/ingredients', homeController.searchByIngredients);
router.post('/search/title', homeController.searchByTitle);
router.post('/categories', homeController.getCategories); // Mendapatkan kategori resep
router.get('/discover', homeController.discover); // Menemukan 5 resep terbaru

// Error handling middleware
router.use((err, req, res, next) => {
    console.error(err);
    res.status(500).send({ error: 'Something went wrong' });
});

module.exports = router;