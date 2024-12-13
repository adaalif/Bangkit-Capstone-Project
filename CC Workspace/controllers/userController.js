const db = require('../config/firestoreDb.js');

// Fungsi untuk mengambil profil pengguna
const getProfile = async (req, res) => {
  try {
    const userId = req.user?.id || req.body.userId; // Hanya menggunakan ID dari token JWT atau body
    if (!userId) {
      return res.status(401).json({ message: 'Unauthorized: User ID is required' });
    }

    const userDoc = await db.collection('users').doc(userId).get();
    if (!userDoc.exists) {
      return res.status(404).json({ message: 'User not found' });
    }

    const userData = userDoc.data();
    // Hindari mengirim data sensitif
    delete userData.password;
    delete userData.securityQuestions;

    res.status(200).json({
      success: true,
      data: userData
    });
  } catch (error) {
    console.error('Error getting profile:', error);
    res.status(500).json({ success: false, message: 'Internal server error' });
  }
};

// Fungsi untuk mengupdate profil pengguna
const updateUserProfile = async (req, res) => {
  try {
    const userId = req.user?.id || req.body.userId;
    const updateData = {};
    const allowedFields = ['name', 'email', 'phone', 'address', 'profilePicUrl'];

    if (!userId) {
      return res.status(401).json({ success: false, message: 'Unauthorized: User ID is required' });
    }

    // Validasi dan sanitasi input
    allowedFields.forEach(field => {
      if (req.body[field]) {
        updateData[field] = req.body[field].trim();
      }
    });

    if (Object.keys(updateData).length === 0) {
      return res.status(400).json({ 
        success: false, 
        message: 'At least one valid field must be provided for update' 
      });
    }

    // Validasi email jika ada
    if (updateData.email) {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(updateData.email)) {
        return res.status(400).json({ 
          success: false, 
          message: 'Invalid email format' 
        });
      }
    }

    // Validasi URL gambar jika ada
    if (updateData.profilePicUrl) {
      const urlRegex = /^https?:\/\/.+\.(jpg|jpeg|png|gif|webp)$/i;
      if (!urlRegex.test(updateData.profilePicUrl)) {
        return res.status(400).json({ 
          success: false, 
          message: 'Invalid image URL format' 
        });
      }
    }

    const userRef = db.collection('users').doc(userId);
    const userDoc = await userRef.get();
    
    if (!userDoc.exists) {
      return res.status(404).json({ success: false, message: 'User not found' });
    }

    await userRef.update(updateData);

    res.status(200).json({
      success: true,
      message: 'Profile updated successfully',
      data: updateData
    });
  } catch (error) {
    console.error('Error updating profile:', error);
    res.status(500).json({ success: false, message: 'Internal server error' });
  }
};

// Fungsi untuk mendapatkan tema
const getTheme = async (req, res) => {
  try {
    const themeDoc = await db.collection('settings').doc('theme').get();
    if (!themeDoc.exists) {
      return res.status(404).json({ message: 'Theme not found.' });
    }

    res.status(200).json({
      message: 'Get theme successfully',
      theme: themeDoc.data(),
    });
  } catch (error) {
    console.error('Error getting theme:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
};

// Fungsi tentang pengembang
const about = (req, res) => {
  res.status(200).json({
    message: 'This application was built by XYZ Developer. Contact us at xyz@example.com.',
  });
};

const updateProfilePicture = async (req, res) => {
  try {
    const userId = req.body.userId;
    const { profilePicUrl } = req.body;

    // Validasi userId
    if (!userId) {
      return res.status(401).json({ 
        success: false, 
        message: 'Unauthorized: User ID is required' 
      });
    }

    // Validasi profilePicUrl
    if (!profilePicUrl || typeof profilePicUrl !== 'string') {
      return res.status(400).json({ 
        success: false, 
        message: 'Valid profile picture URL is required' 
      });
    }

    // Validasi format URL gambar menggunakan regex
    const urlRegex = /^https?:\/\/.+\.(jpg|jpeg|png|gif|webp)$/i;
    if (!urlRegex.test(profilePicUrl)) {
      return res.status(400).json({ 
        success: false, 
        message: 'Invalid image URL format. URL must end with .jpg, .jpeg, .png, .gif, or .webp' 
      });
    }

    // Cek apakah user exists
    const userRef = db.collection('users').doc(userId);
    const userDoc = await userRef.get();
    
    if (!userDoc.exists) {
      return res.status(404).json({ 
        success: false, 
        message: 'User not found' 
      });
    }

    // Update foto profil
    await userRef.update({ 
      profilePicUrl,
      updatedAt: new Date().toISOString()
    });

    // Ambil data user terbaru
    const updatedUserDoc = await userRef.get();
    const userData = updatedUserDoc.data();

    res.status(200).json({
      success: true,
      message: 'Profile picture updated successfully',
      data: {
        profilePicUrl: userData.profilePicUrl,
        updatedAt: userData.updatedAt
      }
    });

  } catch (error) {
    console.error('Error updating profile picture:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Internal server error' 
    });
  }
};

// Export fungsi
module.exports = {
  getProfile,
  updateUserProfile,
  getTheme,
  about,
  updateProfilePicture
};
