class User {
    constructor(id, email, name, weight, height, activityFrequency, theme, profilePicUrl) {
      this.id = id; // ID pengguna
      this.email = email; // Email pengguna
      this.name = name; // Nama pengguna
      this.weight = weight; // Berat badan pengguna
      this.height = height; // Tinggi badan pengguna
      this.activityFrequency = activityFrequency; // Frekuensi aktivitas pengguna
      this.theme = theme; // Tema aplikasi
      this.profilePicUrl = profilePicUrl; // URL gambar profil
    }
  
    // Simpan atau update profil pengguna
    async save(db) {
      const userRef = db.collection('users').doc(this.id);
      await userRef.set({
        email: this.email,
        name: this.name,
        weight: this.weight,
        height: this.height,
        activityFrequency: this.activityFrequency,
        theme: this.theme,
        profilePicUrl: this.profilePicUrl,
      });
    }
  
    // Ambil profil pengguna berdasarkan ID
    static async fetchProfile(db, id) {
      const doc = await db.collection('users').doc(id).get();
      if (!doc.exists) {
        throw new Error('User not found');
      }
      const data = doc.data();
      return new User(id, data.email, data.name, data.weight, data.height, data.activityFrequency, data.theme, data.profilePicUrl);
    }
  
    // Update data pengguna
    static async updateProfile(db, id, updates) {
      await db.collection('users').doc(id).update(updates);
    }
  }
  
  module.exports = User;
  