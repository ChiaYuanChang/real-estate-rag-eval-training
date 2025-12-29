import json
from io import BytesIO

import requests
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer, util


class PropertyAligner:
    def __init__(self):
        print("Loading Multilingual CLIP model...")
        self.text_model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
        self.img_model = SentenceTransformer('clip-ViT-B-32')

    def _download_image(self, url):
        """

        """
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(url, headers=headers, stream=True, timeout=5)
            if resp.status_code == 200:
                return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            pass
        return None

    def _preprocess_features(self, feature_list):
        texts = []
        for item in feature_list:
            # Combine room + tags into a strong textual feature.
            # example: "客廳(一樓): 挑高空間, 規則格局, 採光中等"
            desc = f"{item['room']}: {', '.join(item['tag_list'])}"
            texts.append(desc)
        return texts

    def align(self, picture_urls, feature_list, threshold=0.20):
        # 1. Prepare images
        print(f"Downloading {len(picture_urls)} images...")
        valid_imgs = []
        valid_urls = []
        for url in picture_urls:
            img = self._download_image(url)
            if img:
                valid_imgs.append(img)
                valid_urls.append(url)

        if not valid_imgs:
            return []

        # Prepare text descriptions
        text_descs = self._preprocess_features(feature_list)

        # Compute embeddings
        print("Encoding features...")
        img_emb = self.img_model.encode(valid_imgs, convert_to_tensor=True)
        text_emb = self.text_model.encode(text_descs, convert_to_tensor=True)

        # Compute similarity matrix (Cosine Similarity)
        # Shape: [N_images, M_texts]
        similarity_matrix = util.cos_sim(img_emb, text_emb).cpu().numpy()

        # Hungarian algorithm for global optimal matching
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

        # Format results
        aligned_results = []

        # Track matched indices to find unmatched items later
        matched_img_indices = set(row_ind)
        matched_text_indices = set(col_ind)

        # Handle matched pairs
        for r, c in zip(row_ind, col_ind):
            score = float(similarity_matrix[r][c])

            match_data = {
                "type": "matched",
                "image_url": valid_urls[r],
                "feature_index": int(c),
                "room": feature_list[c]['room'],
                "tags": feature_list[c]['tag_list'],
                "similarity_score": score,
                "is_confident": score > threshold
            }
            aligned_results.append(match_data)

        # Handle unmatched images (extra images)
        for i in range(len(valid_imgs)):
            if i not in matched_img_indices:
                aligned_results.append({
                    "type": "unmatched_image",
                    "image_url": valid_urls[i],
                    "similarity_score": 0.0
                })

        # Handle unmatched features (extra features)
        for j in range(len(feature_list)):
            if j not in matched_text_indices:
                aligned_results.append({
                    "type": "unmatched_feature",
                    "room": feature_list[j]['room'],
                    "tags": feature_list[j]['tag_list'],
                    "similarity_score": 0.0
                })

        # Sort results by original feature order for easier human reading.
        # Push unmatched images to the end (feature_index defaults to 999).
        aligned_results.sort(key=lambda x: (x.get('feature_index', 999), x['type']))

        return aligned_results


aligner = PropertyAligner()

data = json.load(open("../../data/cleaned_twhg_with_latlng_and_places/property_011_47351281.json"))
result = aligner.align(data['picture_list'], data['extracted_feature_list'])

print(json.dumps(result, indent=2, ensure_ascii=False))
