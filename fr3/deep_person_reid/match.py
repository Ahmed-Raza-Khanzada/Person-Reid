import numpy as np
import os

def calculate_similarity(face_encoding, known_encodings):
    """
    Calculate similarity between the given face encoding and a list of known face encodings.
    Return the index of the best match and its similarity score.
    """
    similarities = np.linalg.norm(face_encoding - known_encodings, axis=1)
    best_match_index = np.argmin(similarities)
    best_similarity = 1 - similarities[best_match_index]  # Convert distance to similarity
    return best_match_index, best_similarity

def match_face_encoding(face_encoding, known_encodings_list):
    """
    Match the given face encoding with a list of known face encodings and return the person with the highest probability.
    """
    best_match_index = None
    best_similarity = -1  # Initialize with a low value
    best_person_id = None

    for person_id, known_encodings in known_encodings_list.items():
        match_index, similarity = calculate_similarity(face_encoding, known_encodings)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match_index = match_index
            best_person_id = person_id

    return best_person_id, best_similarity

# Load known encodings
encodings_directory = 'encodings'
known_encodings_list = {}
for file_name in os.listdir(encodings_directory):
    if file_name.endswith('.npy'):
        person_id = file_name.split('.')[0]
        encodings_array = np.load(os.path.join(encodings_directory, file_name))
        known_encodings_list[person_id] = encodings_array

# Sample face encoding to match
# Replace this with your actual face encoding
face_encoding_to_match = np.random.rand(128)  # Example random face encoding

# Match the face encoding
best_person_id, best_similarity = match_face_encoding(face_encoding_to_match, known_encodings_list)

if best_person_id:
    print(f"Best match: Person ID '{best_person_id}' with similarity {best_similarity}")
else:
    print("No match found")
