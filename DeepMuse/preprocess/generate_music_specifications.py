"""
Generate the music specifications.
Includes:
    key, melody, octave, notes, tempo
"""


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def remove_multiple_keys(d, keys):
    r = dict(d)
    for key in keys:
        del r[key]
    return r

def calc_key(emotion_dict):
    """
    Determine which musical key to use.
    Right now, it is either C-major or C-minor.
    
    :returns: 
    :rtype:   
    """

    net_pos = emotion_dict['positive'] - emotion_dict['negative']
    if net_pos > 0:
        return "c major"
    else:
        return "c minor"


def calc_js(emotion_dict):
    """
    Calculate the lowest and highest density difference between 
    the joy and sadness densities of the novel.
    This is used to determine the octave of the melody.
    """

    return emotion_dict["joy"] - emotion_dict["sadness"]


def calc_as(emotion_dict):
    """
    Calculate the activity score, defined to
    be the difference between the average density of the active 
    emotions (anger and joy) and the average density of the passive
    emotions (sadness).
    This is used to determine tempo.
    """

    activity_score = (emotion_dict["anger"] + emotion_dict["joy"]) / 2
    activity_score = activity_score - emotion_dict["sadness"]
    return activity_score


def calc_min_max_differences(literature_emotions):
    """
    Calculate the lowest and highest density difference between 
    the joy and sadness densities of the novel.
    Calculate the lowest and highest activity score.
    
    :param literature_emotions: 
    :type  literature_emotions: dict 
    :returns: 
    :rtype:   dict
    """

    # joy_sadness_diffs = []
    # activity_scores = []

    # for title, emotion_dict in literature_emotions.items():
    #     joy_sadness_diff = calc_js(emotion_dict)
    #     joy_sadness_diffs.append(joy_sadness_diff)
    #     activity_score = calc_as(emotion_dict)
    #     activity_scores.append(activity_score)

    # max_JS = max(joy_sadness_diffs)
    # min_JS = min(joy_sadness_diffs)
    # max_AS = max(activity_scores)
    # min_AS = min(activity_scores)

    # diff_dict = dict()
    # diff_dict["max_JS"] = max_JS
    # diff_dict["min_JS"] = min_JS
    # diff_dict["max_AS"] = max_AS
    # diff_dict["min_AS"] = min_AS

    # numbers are from Hannah's paper
    diff_dict = {
        "max_JS": 0.0080,
        "min_JS": -0.0080,
        "max_AS": 0.017,
        "min_AS": -0.002,
    }
    return diff_dict


def calc_overall_octave(emotion_dict, js_min, js_max):
    """
    Calculate the overall octave.
    
    :returns: octave
    :rtype:   int
    """

    js = calc_js(emotion_dict)
    octave = (js - js_min) * (6 - 4)
    octave = octave / (js_max - js_min)
    octave = int(round(octave))
    octave = 4 + octave

    return octave


def calc_emotion_octave(o_oct, emotion_dict):
    """
    Calculate the secondary octaves that
    are based on the emotion.
    
    :param o_oct: overall octave value
    :type  o_oct: int
    :param emotion_dict: dict mapping emotion to emotion counts
                         of entire novel
    :type  emotion_dict: dict
    :returns: 
    :rtype:   (int, int)
    """

    emotion_dict = remove_multiple_keys(
        emotion_dict, 
        ["positive", "negative"]
    )

    prevalent_emotion1 = max(emotion_dict, key=emotion_dict.get)
    emotion_dict = removekey(emotion_dict, prevalent_emotion1)

    prevalent_emotion2 = max(emotion_dict, key=emotion_dict.get)

    def overall_to_emotion(overall, emotion):
        """
        Calculate the emotional octaves from the overall octave.
        """

        if emotion in {"joy", "trust"}:
            return overall + 1
        elif emotion in {"anger", "fear", "sadness", "disgust"}:
            return overall - 1
        else:
            return overall

    e_oct1 = overall_to_emotion(o_oct, prevalent_emotion1)
    e_oct2 = overall_to_emotion(o_oct, prevalent_emotion2)

    emo_octave_dict = {
        prevalent_emotion1: e_oct1,
        prevalent_emotion2: e_oct2
    }

    return emo_octave_dict


def calc_tempo(emotion_dict, act_min, act_max):
    """
    Calculate the tempo for the section.
    
    :returns:
    :rtype:   int
    """

    act = calc_as(emotion_dict)
    tempo = (act - act_min) * (180 - 40)
    tempo = tempo / (act_max - act_min)
    return int(round(tempo + 40))


def calc_num_of_notes(emo_density, min_density, max_density):
    """
    Calculate the number of notes for a measure.
    
    Process:
        Split the interval between maximum emotional density and 
        minimum density into 5 equal parts (5 for the number of choices
        for note duration: whole, half, quarter, eighth, sixteenth).
        If emotional density lies in the first interval, it is mapped to whole.
        Etc. until the last interval, where it is mapped to sixteenth
        
    :param max_density: maximum emotion density of the entire novel.
                        Can be of any emotion or of all emotions.
    :type  max_density: float
    :returns: number of notes in the measure
    :rtype:   int
    """

    interval = (max_density - min_density) / 5

    for i in range(5):
        if min_density + (i * interval) <= emo_density <= min_density + (
            (i + 1) * interval):
            return int(2 ** i)


def calc_pitch(emo_density, min_density, max_density, key):
    """
    Calculate the pitch.
    """

    if key == "c major":
        pitches = ["C", "G", "E", "A", "D", "F", "B"]
    elif key == "c minor":
        pitches = ["C", "G", "D#", "G#", "D", "F", "A#"]
    else:
        raise ValueError("Not an available key")

    total_pitches = len(pitches)
    interval = (max_density - min_density) / total_pitches

    for i in range(total_pitches):
        if min_density + (i * interval) <= emo_density <= min_density + (
            (i + 1) * interval):
            return pitches[i]