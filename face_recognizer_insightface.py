import os
import cv2
import io
import datetime
import numpy as np
import psycopg2
from PIL import Image, UnidentifiedImageError
from psycopg2 import Binary
from insightface.app import FaceAnalysis
from config import POSTGRES_CONFIG

class FaceRecognizer:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=POSTGRES_CONFIG['host'],
            port=POSTGRES_CONFIG['port'],
            dbname=POSTGRES_CONFIG['dbname'],
            user=POSTGRES_CONFIG['user'],
            password=POSTGRES_CONFIG['password']
        )
        self.cursor = self.conn.cursor()
        self._create_table()

        self.save_dir = "recognized_faces"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

    def _create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id SERIAL PRIMARY KEY,
                embedding BYTEA,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    def compare_embeddings(self, emb1, emb2):
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(emb1, emb2) / (norm1 * norm2)

    def recognize_face(self, embedding, threshold=0.7):
        self.cursor.execute("SELECT id, embedding FROM faces")
        for face_id, emb_blob in self.cursor.fetchall():
            if isinstance(emb_blob, memoryview):
                emb_blob = emb_blob.tobytes()
            stored_emb = np.frombuffer(emb_blob, dtype=np.float32)
            similarity = self.compare_embeddings(embedding, stored_emb)
            if similarity >= threshold:
                return face_id, similarity
        return None, None

    def store_embedding(self, embedding, file_path):
        self.cursor.execute(
            "INSERT INTO faces (embedding, file_path) VALUES (%s, %s) RETURNING id",
            (Binary(embedding.tobytes()), file_path)
        )
        new_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return new_id

    def save_face(self, face_img, face_id, new_face):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        label = "new" if new_face else "known"
        filename = f"face_{face_id}_{label}_{timestamp}.jpg"
        path = os.path.join(self.save_dir, filename)
        cv2.imwrite(path, face_img)
        print(f"Лицо сохранено в: {path}")
        return path

    def process_image(self, image_input, threshold=0.7):
        if isinstance(image_input, bytes):
            try:
                image_stream = io.BytesIO(image_input)
                pil_img = Image.open(image_stream).convert("RGB")
                image = np.array(pil_img)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except UnidentifiedImageError:
                raise ValueError("Некорректный формат изображения: не удалось декодировать JPEG")
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise ValueError("Unsupported image input type. Expected bytes or np.ndarray.")

        faces = self.face_app.get(image)
        if not faces:
            return []

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = map(int, face.bbox)
        face_img = image[y1:y2, x1:x2]

        embedding = face.embedding.astype(np.float32)
        face_id, similarity = self.recognize_face(embedding, threshold)

        if face_id is not None:
            new_face = False
            file_path = self.save_face(face_img, face_id, new_face)
        else:
            temp_id = "temp"
            file_path = self.save_face(face_img, temp_id, new_face=True)
            face_id = self.store_embedding(embedding, file_path)
            new_path = os.path.join(self.save_dir, f"face_{face_id}_new_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            os.rename(file_path, new_path)
            file_path = new_path
            self.cursor.execute("UPDATE faces SET file_path = %s WHERE id = %s", (file_path, face_id))
            self.conn.commit()

        return [{
                    "embedding": embedding,
                    "face": face_img,
                    "face_id": str(face_id)
                }]


    def close(self):
        self.cursor.close()
        self.conn.close()
