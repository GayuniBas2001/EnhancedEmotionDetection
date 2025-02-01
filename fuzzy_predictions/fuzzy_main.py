from fuzzy_detection import display_image, draw_landmarks_on_image, plot_face_blendshapes_bar_graph
from fuzzy_inputs import calculate_fuzzy_inputs
from fuzzy_logic import fuzzy_rules, evaluate_fuzzy_rules

# Read the input image
image_path = "content/angry.jpg"

display_image(image_path)

detection_result = draw_landmarks_on_image(image_path=image_path)
fuzzy_input_results = calculate_fuzzy_inputs(detection_result.face_blendshapes[0])

#print values
print(f"Mouth Curvature Score: {fuzzy_input_results[0]}")
print(f"Eye Openness Score: {fuzzy_input_results[1]}")
print(f"Eyebrow Position Score: {fuzzy_input_results[2]}")
print(f"Jaw Position Score: {fuzzy_input_results[3]}")
print(f"Nose Sneer Score: {fuzzy_input_results[4]}")

emotion_sim, emotion_graph = fuzzy_rules()
emotion_result = evaluate_fuzzy_rules(emotion_sim, emotion_graph,  *fuzzy_input_results)


