const express = require('express');
const router = express.Router();
const favoriteController = require('../controllers/favoriteController');

// Mendapatkan daftar semua favorit
router.get('/', favoriteController.getFavorites);

// Menambahkan resep ke favorit
router.post('/addfavorite', favoriteController.addFavorite);

// Menghapus resep dari favorit
router.delete('/deletefavorite', favoriteController.removeFavorite);

// Mendapatkan kategori dari daftar favorit
router.post('/categories', favoriteController.getFavoriteCategories);

module.exports = router;
