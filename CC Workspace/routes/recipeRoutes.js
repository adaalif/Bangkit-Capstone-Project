const express = require('express');
const router = express.Router();
const { getAllRecipes, getRecipeById } = require('../controllers/recipeController');

// Recipe endpoints
router.get('/', getAllRecipes);
router.get('/:id', getRecipeById); 

module.exports = router;
