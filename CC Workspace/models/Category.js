class Category {
    constructor(id, name, description, imageUrl) {
      this.id = id; // ID kategori
      this.name = name; // Nama kategori
      this.description = description; // Deskripsi kategori
      this.imageUrl = imageUrl; // URL gambar kategori
    }
  
    // Simpan atau update kategori
    async save(db) {
      const categoryRef = db.collection('categories').doc(this.id || undefined);
      await categoryRef.set({
        name: this.name,
        description: this.description,
        imageUrl: this.imageUrl,
      });
      this.id = categoryRef.id;
    }
  
    // Ambil semua kategori
    static async fetchAll(db) {
      const snapshot = await db.collection('categories').get();
      return snapshot.docs.map(doc => new Category(
        doc.id,
        doc.data().name,
        doc.data().description,
        doc.data().imageUrl
      ));
    }
  
    // Ambil kategori berdasarkan ID
    static async findById(db, id) {
      const doc = await db.collection('categories').doc(id).get();
      if (!doc.exists) {
        throw new Error('Category not found');
      }
      const data = doc.data();
      return new Category(id, data.name, data.description, data.imageUrl);
    }
  }
  
  module.exports = Category;
  