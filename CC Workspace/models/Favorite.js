class Favorite {
    constructor(userId, recipeId) {
      this.userId = userId; // ID pengguna
      this.recipeId = recipeId; // ID resep
    }
  
    // Tambahkan resep ke favorit
    static async addFavorite(db, userId, recipeId) {
      const favoriteRef = db.collection('favorites').doc(userId);
      await favoriteRef.set({ [recipeId]: true }, { merge: true });
    }
  
    // Hapus resep dari favorit
    static async removeFavorite(db, userId, recipeId) {
      const favoriteRef = db.collection('favorites').doc(userId);
      await favoriteRef.update({
        [recipeId]: admin.firestore.FieldValue.delete(),
      });
    }
  
    // Ambil semua favorit pengguna
    static async fetchAll(db, userId) {
      const doc = await db.collection('favorites').doc(userId).get();
      if (!doc.exists) {
        return [];
      }
      const data = doc.data();
      return Object.keys(data).map(recipeId => recipeId);
    }
  }
  
  module.exports = Favorite;
  