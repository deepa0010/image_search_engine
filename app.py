
# ###444save_encoding Final
# import streamlit as st
# from PIL import Image
# import face_recognition
# import pickle
# import os

# def load_encodings(file_path):
#     with open(file_path, 'rb') as file:
#         return pickle.load(file)

# def load_image(image_path):
#     return face_recognition.load_image_file(image_path)

# def extract_face_encoding(image):
#     face_locations = face_recognition.face_locations(image)
#     if len(face_locations) == 0:
#         return None
#     face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
#     return face_encodings

# def find_similar_faces(person_encodings, group_folder):
#     matched_images = []
#     for img_name in os.listdir(group_folder):
#         img_path = os.path.join(group_folder, img_name)
#         image = load_image(img_path)
#         group_encodings = extract_face_encoding(image)
#         if group_encodings:
#             for group_encoding in group_encodings:
#                 matches = face_recognition.compare_faces(person_encodings, group_encoding)
#                 if True in matches:
#                     matched_images.append(img_name)
#                     break
#     return matched_images

# st.title("Wedding Photo Finder")

# # Select a person from the 'person' folder
# person_folder = "dataset/person"
# people = os.listdir(person_folder)
# selected_person = st.selectbox("Select a person:", people)

# # Load all images for the selected person
# person_images_folder = os.path.join(person_folder, selected_person)
# person_image_paths = [os.path.join(person_images_folder, img) for img in os.listdir(person_images_folder)]

# # Display the selected person's images
# st.write(f"Selected Person: {selected_person}")
# for person_image_path in person_image_paths:
#     st.image(person_image_path, caption=f"{selected_person}", use_column_width=True)

# # Load the person's encodings
# encodings_file = f'{selected_person}_encodings.pkl'
# if os.path.exists(encodings_file):
#     person_encodings = load_encodings(encodings_file)
# else:
#     st.error(f"Encodings file not found: {encodings_file}")
#     person_encodings = None

# # Find and display matching group photos
# if person_encodings and st.button("Find Images"):
#     group_folder = "dataset/others"
    
#     matched_images = find_similar_faces(person_encodings, group_folder)
    
#     # Display the results
#     st.write(f"Found {len(matched_images)} images with the person of interest:")
#     for img_name in matched_images:
#         img_path = os.path.join(group_folder, img_name)
#         img = Image.open(img_path)
#         st.image(img, caption=os.path.basename(img_path), use_column_width=True)


# import streamlit as st
# from PIL import Image
# import face_recognition
# import pickle
# import os

# def load_encodings(file_path):
#     with open(file_path, 'rb') as file:
#         return pickle.load(file)

# def load_image(image_path):
#     return face_recognition.load_image_file(image_path)

# def extract_face_encoding(image):
#     face_locations = face_recognition.face_locations(image)
#     if len(face_locations) == 0:
#         return None
#     face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
#     return face_encodings

# def find_similar_faces(person_encodings, group_folder):
#     matched_images = []
#     for img_name in os.listdir(group_folder):
#         img_path = os.path.join(group_folder, img_name)
#         image = load_image(img_path)
#         group_encodings = extract_face_encoding(image)
#         if group_encodings:
#             for group_encoding in group_encodings:
#                 matches = face_recognition.compare_faces(person_encodings, group_encoding)
#                 if True in matches:
#                     matched_images.append(img_path)
#                     break
#     return matched_images

# st.title("Wedding Photo Finder")

# # Select a person from the 'person' folder
# person_folder = "dataset/person"
# people = os.listdir(person_folder)
# selected_person = st.selectbox("Select a person:", people)

# # Load all images for the selected person
# person_images_folder = os.path.join(person_folder, selected_person)
# person_image_paths = [os.path.join(person_images_folder, img) for img in os.listdir(person_images_folder)]

# # Display the selected person's images
# st.write(f"Selected Person: {selected_person}")
# for person_image_path in person_image_paths:
#     st.image(person_image_path, caption=f"{selected_person}", use_column_width=True)

# # Load the person's encodings
# encodings_file = f'{selected_person}_encodings.pkl'
# if os.path.exists(encodings_file):
#     person_encodings = load_encodings(encodings_file)
# else:
#     st.error(f"Encodings file not found: {encodings_file}")
#     person_encodings = None

# # Find and display matching group photos
# if person_encodings and st.button("Find Images"):
#     group_folder = "dataset/others"
    
#     matched_images = find_similar_faces(person_encodings, group_folder)
    
#     # Display the results in a grid layout
#     st.write(f"Found {len(matched_images)} images with the person of interest:")

#     # Define the number of images per row
#     num_images_per_row = 3

#     # Create columns for the grid layout
#     num_rows = (len(matched_images) + num_images_per_row - 1) // num_images_per_row  # Calculate the number of rows needed
#     image_cols = st.columns(num_images_per_row)  # Create columns for layout

#     # Display images in rows and columns
#     for row in range(num_rows):
#         start_index = row * num_images_per_row
#         end_index = start_index + num_images_per_row
#         row_images = matched_images[start_index:end_index]
#         cols = st.columns(num_images_per_row)  # Create new columns for the current row
#         for col, img_path in zip(cols, row_images):
#             with col:
#                 img = Image.open(img_path)
#                 st.image(img, caption=os.path.basename(img_path), use_column_width=True)


import streamlit as st
from PIL import Image
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

def find_similar_faces(person_encodings, group_folder):
    matched_images = []
    for img_name in os.listdir(group_folder):
        img_path = os.path.join(group_folder, img_name)
        image = load_image(img_path)
        group_encodings = extract_face_encoding(image)
        if group_encodings:
            for group_encoding in group_encodings:
                matches = face_recognition.compare_faces(person_encodings, group_encoding)
                if True in matches:
                    matched_images.append(img_path)
                    break
    return matched_images

st.title("Wedding Photo Finder")

# Select a person from the 'person' folder
person_folder = "dataset/person"
people = os.listdir(person_folder)
selected_person = st.selectbox("Select a person:", people)

# Load all images for the selected person
person_images_folder = os.path.join(person_folder, selected_person)
person_image_paths = [os.path.join(person_images_folder, img) for img in os.listdir(person_images_folder)]

# Display the selected person's images
st.write(f"Selected Person: {selected_person}")
for person_image_path in person_image_paths:
    st.image(person_image_path, caption=f"{selected_person}", use_column_width=True)

# Load the person's encodings
encodings_file = f'{selected_person}_encodings.pkl'
if os.path.exists(encodings_file):
    person_encodings = load_encodings(encodings_file)
else:
    st.error(f"Encodings file not found: {encodings_file}")
    person_encodings = None

# Find and display matching group photos
if person_encodings and st.button("Find Images"):
    group_folder = "dataset/others"
    
    matched_images = find_similar_faces(person_encodings, group_folder)
    
    # Display the results in a grid layout
    st.write(f"Found {len(matched_images)} images with the person of interest:")

    # Define the number of images per row
    num_images_per_row = 3

    # Define the column widths as a ratio (e.g., 1:1 for equal width)
    column_widths = [1] * num_images_per_row

    # Create columns for the grid layout
    num_rows = (len(matched_images) + num_images_per_row - 1) // num_images_per_row  # Calculate the number of rows needed

    for row in range(num_rows):
        start_index = row * num_images_per_row
        end_index = start_index + num_images_per_row
        row_images = matched_images[start_index:end_index]
        
        # Create columns for the current row
        cols = st.columns(column_widths)  # Use the defined widths

        for col, img_path in zip(cols, row_images):
            with col:
                img = Image.open(img_path)
                st.image(img, caption=os.path.basename(img_path), use_column_width=True)



