class Recipe {
    constructor(id, name, ingredients, imageUrl, calories, steps, category, complexity) {
      this.id = id; // ID resep
      this.name = name; // Nama resep
      this.ingredients = ingredients; // Daftar bahan
      this.imageUrl = imageUrl; // URL gambar resep
      this.calories = calories; // Jumlah kalori
      this.steps = steps; // Langkah memasak
      this.category = category; // Kategori resep
      this.complexity = complexity; // Tingkat kesulitan
    }
  
    // Simpan atau update resep
    async save(db) {
      const recipeRef = db.collection('recipes').doc(this.id || undefined);
      await recipeRef.set({
        name: this.name,
        ingredients: this.ingredients,
        imageUrl: this.imageUrl,
        calories: this.calories,
        steps: this.steps,
        category: this.category,
        complexity: this.complexity,
      });
      this.id = recipeRef.id;
    }
  
    // Ambil semua resep
    static async fetchAll(db) {
      const snapshot = await db.collection('recipes').get();
      return snapshot.docs.map(doc => new Recipe(
        doc.id,
        doc.data().name,
        doc.data().ingredients,
        doc.data().imageUrl,
        doc.data().calories,
        doc.data().steps,
        doc.data().category,
        doc.data().complexity
      ));
    }
  
    // Ambil resep berdasarkan ID
    static async findById(db, id) {
      const doc = await db.collection('recipes').doc(id).get();
      if (!doc.exists) {
        throw new Error('Recipe not found');
      }
      const data = doc.data();
      return new Recipe(id, data.name, data.ingredients, data.imageUrl, data.calories, data.steps, data.category, data.complexity);
    }
  
    // Cari resep berdasarkan nama
    static async searchByName(db, query) {
      const snapshot = await db.collection('recipes')
        .where('name', '>=', query)
        .where('name', '<=', query + '\uf8ff')
        .get();
      return snapshot.docs.map(doc => new Recipe(
        doc.id,
        doc.data().name,
        doc.data().ingredients,
        doc.data().imageUrl,
        doc.data().calories,
        doc.data().steps,
        doc.data().category,
        doc.data().complexity
      ));
    }
  }
  
  module.exports = Recipe;
  