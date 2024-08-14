# import os
# from PIL import Image
# import numpy as np
# import torch
# import torchvision
# from torchvision import transforms

# images = os.listdir("./data/")

# model = torchvision.models.resnet18(weights="DEFAULT")

# all_names = []
# all_vecs = None
# model.eval()
# root = "./data/"

# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# activation = {}
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#     return hook

# try:
#     model.avgpool.register_forward_hook(get_activation('avgpool'))
# except Exception as e:
#     print(f"Error registering hook: {e}")

# with torch.no_grad():
#     for i, file in enumerate(images):
#         try:
#             img = Image.open(root + file)
#             img = transform(img)
#             img = img.unsqueeze(0)  # Add batch dimension
#             out = model(img)
#             vec = activation['avgpool'].numpy().squeeze()[None, ...]
#             if all_vecs is None:
#                 all_vecs = vec
#             else:
#                 all_vecs = np.vstack([all_vecs, vec])
#             all_names.append(file)
#         except Exception as e:
#             print(f"Error processing {file}: {e}")
#             continue
#         if i % 100 == 0 and i != 0:
#             print(i, "done")

# np.save("all_vecs.npy", all_vecs)
# np.save("all_names.npy", np.array(all_names, dtype=str))






# import os
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# import tensorflow_hub as hub
# from sklearn.preprocessing import normalize

# # Load FaceNet model
# model = hub.load("https://tfhub.dev/google/facenet/2")

# def preprocess_image(image_path):
#     img = Image.open(image_path).resize((160, 160))
#     img = np.array(img) / 255.0
#     img = img[np.newaxis, ...]
#     return img

# def get_image_embedding(image_path):
#     img = preprocess_image(image_path)
#     embedding = model(img)
#     return embedding.numpy().flatten()

# # Load dataset
# image_dir = "./images/"
# all_names = []
# all_vecs = []

# for file_name in os.listdir(image_dir):
#     try:
#         img_path = os.path.join(image_dir, file_name)
#         embedding = get_image_embedding(img_path)
#         all_vecs.append(embedding)
#         all_names.append(file_name)
#     except Exception as e:
#         print(f"Error processing {file_name}: {e}")

# all_vecs = np.array(all_vecs)
# all_vecs = normalize(all_vecs)  # Normalize embeddings

# np.save("all_vecs.npy", all_vecs)
# np.save("all_names.npy", np.array(all_names))

######### APP.PY 111
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.optimizers import Adam
# import numpy as np
# from tensorflow.keras.preprocessing import image

# # Load the pre-trained VGG16 model
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # Freeze the base model
# for layer in base_model.layers:
#     layer.trainable = False

# # Add custom layers on top
# x = Flatten()(base_model.output)
# x = Dense(128, activation='relu')(x)
# x = Dense(1, activation='sigmoid')(x)  # Binary classification: person of interest or not

# model = Model(inputs=base_model.input, outputs=x)

# # Compile the model
# model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# # Data augmentation and generators
# train_datagen = ImageDataGenerator(rescale=0.2, horizontal_flip=True)
# train_generator = train_datagen.flow_from_directory(
#     'dataset/',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary'
# )

# # Train the model
# model.fit(train_generator, epochs=10)

# # Save the trained model
# model.save('person_of_interest_model.h5')

# # Function to predict if the person of interest is in an image
# def is_person_of_interest(model, img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0

#     prediction = model.predict(img_array)
#     return prediction[0][0] > 0.5




####final
import face_recognition
import pickle
import os

def load_encodings(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def load_image(image_path):
    return face_recognition.load_image_file(image_path)

def extract_face_encoding(image):
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        return None
    face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
    return face_encodings

def find_similar_faces(person_encodings_dict, group_folder):
    matched_images = {person: [] for person in person_encodings_dict}
    
    for img_name in os.listdir(group_folder):
        img_path = os.path.join(group_folder, img_name)
        print(f"Analyzing {img_path}...")
        image = load_image(img_path)
        group_encodings = extract_face_encoding(image)
        
        if group_encodings:
            for person_name, person_encodings in person_encodings_dict.items():
                for group_encoding in group_encodings:
                    matches = face_recognition.compare_faces(person_encodings, group_encoding)
                    if True in matches:
                        matched_images[person_name].append(img_name)
                        break
        else:
            print(f"No faces detected in {img_path}")
    return matched_images

def main():
    # Load encodings for all persons in the 'person' folder
    person_folder = "dataset/person"
    encodings_dict = {}

    for person_folder_name in os.listdir(person_folder):
        person_folder_path = os.path.join(person_folder, person_folder_name)
        if os.path.isdir(person_folder_path):
            encodings_file = f'{person_folder_name}_encodings.pkl'
            if os.path.exists(encodings_file):
                encodings_dict[person_folder_name] = load_encodings(encodings_file)
            else:
                print(f"Encodings file not found for {person_folder_name}: {encodings_file}")

    # Specify the group folder
    group_folder = "dataset/others"
    
    # Find and display matching images
    matched_images = find_similar_faces(encodings_dict, group_folder)
    
    for person, images in matched_images.items():
        print(f"Found {len(images)} images with {person}:")
        for img_name in images:
            print(img_name)

if __name__ == "__main__":
    main()




