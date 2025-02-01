# Calculate the fuzzy inputs for the fuzzy controller

def get_blendshape_score(blendshapes, name):
    for blendshape in blendshapes:
        if blendshape.category_name == name:
            return min(blendshape.score, 0.7)  # Clamp scores to a max of 0.7
    return 0.0  # Default to 0 if the blendshape is not found

# Normalize scores to fit within [0, 0.7]
def normalize_score(score):
    return score / 0.7

def calculate_fuzzy_inputs(blendshapes):

    # Derive necessary parameters
    mouth_curvature_score = normalize_score(
        (
            get_blendshape_score(blendshapes, 'mouthSmileLeft')
            + get_blendshape_score(blendshapes, 'mouthSmileRight')
            - get_blendshape_score(blendshapes, 'mouthFrownLeft')
            - get_blendshape_score(blendshapes, 'mouthFrownRight')
        ) / 2
    )

    eye_openness_score = normalize_score(
        (
            get_blendshape_score(blendshapes, 'eyeWideLeft')
            + get_blendshape_score(blendshapes, 'eyeWideRight')
        ) / 2
    )

    eyebrow_position_score = normalize_score(
        get_blendshape_score(blendshapes, 'browInnerUp')
        - (
            get_blendshape_score(blendshapes, 'browDownLeft') 
            + get_blendshape_score(blendshapes, 'browDownRight')
        ) / 2
    )

    jaw_position_score = normalize_score(
        get_blendshape_score(blendshapes, 'jawOpen')
    )

    nose_sneer_score = normalize_score(
        (
            get_blendshape_score(blendshapes, 'noseSneerLeft') 
            + get_blendshape_score(blendshapes, 'noseSneerRight')
        ) / 2
    )

    # #print values
    # print(f"Mouth Curvature Score: {mouth_curvature_score}")
    # print(f"Eye Openness Score: {eye_openness_score}")
    # print(f"Eyebrow Position Score: {eyebrow_position_score}")
    # print(f"Jaw Position Score: {jaw_position_score}")
    # print(f"Nose Sneer Score: {nose_sneer_score}")

    # # Pass these normalized parameters to the fuzzy logic system
    # emotion_sim.input['mouth_curvature'] = mouth_curvature_score
    # emotion_sim.input['eye_openness'] = eye_openness_score
    # emotion_sim.input['eyebrow_position'] = eyebrow_position_score
    # emotion_sim.input['jaw_position'] = jaw_position_score
    # emotion_sim.input['nose_sneer'] = nose_sneer_score

    return [mouth_curvature_score, eye_openness_score, eyebrow_position_score, jaw_position_score, nose_sneer_score]