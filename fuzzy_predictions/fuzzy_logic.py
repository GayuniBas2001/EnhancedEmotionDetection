import numpy as np
import skfuzzy as fuzz
from skfuzzy.control import Antecedent, Consequent, Rule, ControlSystem, ControlSystemSimulation

# Define Membership Functions
def fuzzy_rules():

    mouth_curvature = Antecedent(np.arange(-1, 1.1, 0.1), 'mouth_curvature')
    eye_openness = Antecedent(np.arange(0, 1.1, 0.1), 'eye_openness')
    eyebrow_position = Antecedent(np.arange(-1, 1.1, 0.1), 'eyebrow_position')
    jaw_position = Antecedent(np.arange(0, 1.1, 0.1), 'jaw_position')
    nose_sneer = Antecedent(np.arange(0, 1.1, 0.1), 'nose_sneer')

    # Define the output variable
    emotion = Consequent(np.arange(0, 1.1, 0.1), 'emotion')

    # Membership functions for mouth curvature
    mouth_curvature['frown'] = fuzz.trimf(mouth_curvature.universe, [-1, -0.5, 0])
    mouth_curvature['neutral'] = fuzz.trimf(mouth_curvature.universe, [-0.2, 0, 0.2])
    mouth_curvature['smile'] = fuzz.trimf(mouth_curvature.universe, [0, 0.5, 1])

    # Membership functions for eye openness
    eye_openness['narrow'] = fuzz.trimf(eye_openness.universe, [0, 0.2, 0.5])
    eye_openness['normal'] = fuzz.trimf(eye_openness.universe, [0.4, 0.6, 0.8])
    eye_openness['wide'] = fuzz.trimf(eye_openness.universe, [0.7, 1, 1])

    # Membership functions for eyebrow position
    eyebrow_position['lowered'] = fuzz.trimf(eyebrow_position.universe, [-1, -0.5, 0])
    eyebrow_position['neutral'] = fuzz.trimf(eyebrow_position.universe, [-0.2, 0, 0.2])
    eyebrow_position['raised'] = fuzz.trimf(eyebrow_position.universe, [0, 0.5, 1])

    # Membership functions for jaw position
    jaw_position['closed'] = fuzz.trimf(jaw_position.universe, [0, 0.2, 0.5])
    jaw_position['slightly_open'] = fuzz.trimf(jaw_position.universe, [0.3, 0.5, 0.7])
    jaw_position['wide_open'] = fuzz.trimf(jaw_position.universe, [0.6, 1, 1])

    # Membership functions for nose sneer
    nose_sneer['none'] = fuzz.trimf(nose_sneer.universe, [0, 0.2, 0.4])
    nose_sneer['moderate'] = fuzz.trimf(nose_sneer.universe, [0.3, 0.5, 0.7])
    nose_sneer['strong'] = fuzz.trimf(nose_sneer.universe, [0.6, 1, 1])

    # Membership functions for emotion
    emotion['happiness'] = fuzz.trimf(emotion.universe, [0.7, 0.9, 1])
    emotion['sadness'] = fuzz.trimf(emotion.universe, [0.1, 0.3, 0.5])
    emotion['surprise'] = fuzz.trimf(emotion.universe, [0.6, 0.8, 1])
    emotion['anger'] = fuzz.trimf(emotion.universe, [0.4, 0.6, 0.8])
    emotion['fear'] = fuzz.trimf(emotion.universe, [0.5, 0.7, 0.9])
    emotion['disgust'] = fuzz.trimf(emotion.universe, [0.2, 0.4, 0.6])

    rules = [
        # Happiness
        Rule(mouth_curvature['smile'] & eye_openness['normal'] & eyebrow_position['neutral'], emotion['happiness']),
        Rule(mouth_curvature['smile'] & eye_openness['wide'] & eyebrow_position['raised'], emotion['happiness']),
        Rule(mouth_curvature['smile'] & eye_openness['narrow'], emotion['happiness']),
        Rule(mouth_curvature['neutral'] & eyebrow_position['neutral'], emotion['happiness']),
        
        # Sadness
        Rule(mouth_curvature['frown'] & eyebrow_position['lowered'] & eye_openness['narrow'], emotion['sadness']),
        Rule(mouth_curvature['frown'] & eye_openness['narrow'] & jaw_position['closed'], emotion['sadness']),
        Rule(mouth_curvature['neutral'] & eyebrow_position['lowered'] & eye_openness['narrow'], emotion['sadness']),
        
        # Surprise
        Rule(eye_openness['wide'] & jaw_position['wide_open'] & eyebrow_position['raised'], emotion['surprise']),
        Rule(eye_openness['wide'] & jaw_position['slightly_open'] & eyebrow_position['neutral'], emotion['surprise']),
        Rule(mouth_curvature['neutral'] & eye_openness['wide'] & eyebrow_position['raised'], emotion['surprise']),
        
        # Anger
        Rule(mouth_curvature['frown'] & (eyebrow_position['lowered']|eyebrow_position['neutral']) & eye_openness['narrow'], emotion['anger']),
        Rule(eyebrow_position['lowered'] & eye_openness['narrow'] & nose_sneer['strong'], emotion['anger']),
        Rule(mouth_curvature['frown'] & eyebrow_position['lowered'] & nose_sneer['moderate'], emotion['anger']),
        Rule(eye_openness['narrow'] & eyebrow_position['lowered'] & jaw_position['closed'], emotion['anger']),
        
        # Fear
        Rule(eye_openness['wide'] & eyebrow_position['raised'] & jaw_position['slightly_open'], emotion['fear']),
        Rule(eye_openness['wide'] & mouth_curvature['neutral'] & eyebrow_position['raised'], emotion['fear']),
        Rule(eye_openness['wide'] & jaw_position['wide_open'] & nose_sneer['moderate'], emotion['fear']),
        
        # Disgust
        Rule(nose_sneer['moderate'] & mouth_curvature['frown'] & jaw_position['closed'], emotion['disgust']),
        Rule(nose_sneer['strong'] & mouth_curvature['neutral'] & jaw_position['slightly_open'], emotion['disgust']),
        Rule(eyebrow_position['lowered'] & nose_sneer['strong'] & eye_openness['narrow'], emotion['disgust']),
        
        # Neutral Expression
        Rule(mouth_curvature['neutral'] & eye_openness['normal'] & eyebrow_position['neutral'], emotion['happiness']),
    ]

    # Create control system and simulation
    emotion_ctrl = ControlSystem(rules)
    emotion_sim = ControlSystemSimulation(emotion_ctrl)

    return emotion_sim, emotion

# evaluate the fuzzy rules
def evaluate_fuzzy_rules(emotion_sim, emotion_graph, mouth_curvature, eye_openness, eyebrow_position, jaw_position, nose_sneer):
    emotion_sim.input['mouth_curvature'] = mouth_curvature
    emotion_sim.input['eye_openness'] = eye_openness
    emotion_sim.input['eyebrow_position'] = eyebrow_position
    emotion_sim.input['jaw_position'] = jaw_position
    emotion_sim.input['nose_sneer'] = nose_sneer

    emotion_sim.compute()

    # view output with graph
    person_emotion=emotion_sim.output['emotion']
    print(f"Emotion Confidence Levels: {person_emotion}")
    emotion_graph.view(sim=emotion_sim)

# ===================================================================================================
    # Extract membership levels of all emotions
    membership_levels = {
        'happiness': fuzz.interp_membership(emotion_graph.universe, emotion_graph['happiness'].mf, person_emotion),
        'sadness': fuzz.interp_membership(emotion_graph.universe, emotion_graph['sadness'].mf, person_emotion),
        'surprise': fuzz.interp_membership(emotion_graph.universe, emotion_graph['surprise'].mf, person_emotion),
        'anger': fuzz.interp_membership(emotion_graph.universe, emotion_graph['anger'].mf, person_emotion),
        'fear': fuzz.interp_membership(emotion_graph.universe, emotion_graph['fear'].mf, person_emotion),
        'disgust': fuzz.interp_membership(emotion_graph.universe, emotion_graph['disgust'].mf, person_emotion),
    }

    # Determine the emotion with the highest membership
    final_emotion = max(membership_levels, key=membership_levels.get)
    final_membership = membership_levels[final_emotion]

    print(f"Final Emotion: {final_emotion} with Membership Level: {final_membership:.2f}")
# ===================================================================================================

    return person_emotion

if __name__ == '__main__':
    print('Fuzzy Logic')
    # emotion_sim = fuzzy_rules()
    # emotion = evaluate_fuzzy_rules(emotion_sim, 0.5, 0.7, 0.3, 0.6, 0.8)
    # print('Emotion: ', emotion)