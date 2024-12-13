const express = require('express');
const router = express.Router();
const authController = require('../controllers/authController');


// Mendefinisikan rute

// Register User
router.post('/register', authController.register);

// Login via email User
router.post('/login/email', authController.loginWithEmailPassword);

// Login via Google
router.post('/login/google', authController.loginWithGoogle);

// Forget password User
router.post('/forgot-password', authController.forgotPassword);

// Logout User
router.post('/logout', authController.logout);

module.exports = router;
