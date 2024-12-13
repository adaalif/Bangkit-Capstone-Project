const express = require('express');
const router = express.Router();
const { 
  getProfile, 
  updateUserProfile, 
  updateProfilePicture,
  getTheme, 
  about 
} = require('../controllers/userController');

// User profile routes
router.post('/profile', getProfile);
router.put('/profile', updateUserProfile);
router.put('/profile/picture', updateProfilePicture);  // Endpoint baru khusus untuk update foto

// Public routes
router.get('/theme', getTheme);
router.get('/about', about);

module.exports = router;
