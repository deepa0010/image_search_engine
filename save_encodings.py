#####final
import face_recognition
import pickle
import os

def load_image(image_path):
    return face_recognition.load_image_file(image_path)

def extract_face_encoding(image):
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        return None
    face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
    return face_encodings

def get_face_encodings_from_folder(folder_path):
    face_encodings = []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        image = load_image(img_path)
        encodings = extract_face_encoding(image)
        if encodings:
            face_encodings.extend(encodings)
    return face_encodings

def save_encodings(encodings, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(encodings, file)

# Save encodings for each person
def save_person_encodings():
    person_folder = "dataset/person"
    people = os.listdir(person_folder)
    
    for person in people:
        person_folder_path = os.path.join(person_folder, person)
        encodings = get_face_encodings_from_folder(person_folder_path)
        save_encodings(encodings, f'{person}_encodings.pkl')

if __name__ == "__main__":
    save_person_encodings()
